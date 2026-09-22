"""Etapa 1 (BrEaST) do Protocolo C: monta o índice do BrEaST no mesmo esquema comum do
BUS-BRA, para ser usado como validação externa (nunca como treino).

Lê apenas os arquivos brutos do BrEaST (a planilha clínica
`BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx` e a pasta de imagens/máscaras) e
escreve `dados_processados/breast/indice_breast.csv`, no MESMO esquema (`COLUNAS`) que
`indexar.py` usa para o BUS-BRA — reaproveitado daqui via import, não copiado à mão, para
os dois índices nunca divergirem silenciosamente.

Este script NUNCA mexe em `indexar.py` nem em `dados_processados/indice.csv` (o índice do
BUS-BRA) — é um arquivo novo, lado a lado, que escreve numa pasta separada
(`dados_processados/breast/`), conforme pedido explicitamente para o Protocolo C: o
BrEaST nunca entra misturado com os arquivos do BUS-BRA.

Particularidades do BrEaST em relação ao BUS-BRA (documentadas aqui, cada uma onde é
tratada no código abaixo):

    - a planilha tem 256 linhas, mas 4 são `Classification == "normal"` — essas linhas
      vêm com `Mask_tumor_filename` vazio (NaN) porque não há lesão para segmentar.
      Descartamos essas 4 ANTES de tentar montar qualquer caminho de máscara (nunca
      tentamos abrir um arquivo de máscara para elas). Sobram 252 linhas usáveis, cada
      uma benigna ou maligna.
    - `paciente` = `CaseID` (cada linha da planilha já é um paciente/caso distinto — não
      há o conceito de duas vistas por paciente como no BUS-BRA).
    - `birads` vem como texto na planilha (`"4a"`, `"4b"`, `"5"`, ...) — mantido como
      texto, sem conversão, igual ao BUS-BRA.
    - NÃO existe coluna de aparelho/fabricante na planilha do BrEaST (conferido: as 21
      colunas da planilha não incluem nada equivalente a `Device` do BUS-BRA). A coluna
      `aparelho` do índice fica vazia (None) para todas as linhas do BrEaST — decisão de
      projeto registrada aqui, não um esquecimento.
    - `largura`/`altura` não existem na planilha (ao contrário do BUS-BRA, que declara
      `Width`/`Height`): lidas da imagem real no disco com PIL.
    - `lado` (esquerda/direita) é um conceito específico do BUS-BRA (duas vistas por
      paciente) que não existe no BrEaST — fica vazio (None).

A partir daqui, nenhum outro script do Protocolo C (`verificar_breast.py`,
`features_breast.py`, ...) volta a abrir a planilha bruta do BrEaST: todos leem só
`indice_breast.csv`.

Uso:
    python src/protocolo_c/indexar_breast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

# Reaproveita o esquema comum e as funções utilitárias do BUS-BRA — pacote
# `src/comum/`, ver docstring do módulo. `indexar.py` não é modificado por este
# import.
#
# Desde a reorganização de src/ por trilha (comum/, protocolo_ab/, protocolo_c/,
# resnet/), os módulos reaproveitados ficam em outro pacote: `src/` entra no sys.path
# para que o(s) import(s) abaixo funcionem rodando o script direto, de qualquer
# diretório. Só muda onde o Python procura o módulo — nenhuma lógica é alterada.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from comum.indexar import COLUNAS, exigir, sha1_do_arquivo

# ---------------------------------------------------------------------------
# Caminhos — tudo derivado da posição deste arquivo (ver indexar.py).
# ---------------------------------------------------------------------------

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

BREAST = Path("01-datasets/BrEaST-Lesions-USG")
# Nome real da pasta de imagens, conferido no disco antes de assumir (não se chama
# "Images/" como no BUS-BRA):
BREAST_IMAGENS = BREAST / "BrEaST-Lesions_USG-images_and_masks"

CLINICO = RAIZ_TCC / BREAST / "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"

SAIDA = RAIZ_CODIGO / "dados_processados" / "breast" / "indice_breast.csv"

# Tradução do rótulo — mesma ideia do ROTULO_BUSBRA em indexar.py: também serve de lista
# fechada de valores aceitos (fora "normal", que é descartado antes de chegar aqui).
ROTULO_BREAST = {"benign": 0, "malignant": 1}

# Números conferidos no conjunto bruto (256 linhas na planilha, 4 "normal" descartadas).
LINHAS_BRUTAS_ESPERADAS = 256
NORMAIS_ESPERADOS = 4
LINHAS_ESPERADAS = 252
PACIENTES_ESPERADOS = 252

# Colunas consideradas críticas para a checagem de "nenhuma vazia" — `aparelho` e `lado`
# ficam de fora de propósito, porque são vazias por decisão de projeto (ver docstring),
# não por erro de montagem.
COLUNAS_CRITICAS = ["base", "id", "paciente", "imagem", "mascara", "rotulo", "birads", "sha1"]


def tamanho_da_imagem(caminho: Path) -> tuple[int, int]:
    """(largura, altura) lidos do arquivo real — a planilha do BrEaST não declara essas
    colunas (ao contrário do bus_data.csv, que declara Width/Height)."""
    with Image.open(caminho) as img:
        return img.size


def ler_breast() -> pd.DataFrame:
    """Lê a planilha clínica bruta do BrEaST e devolve a tabela já no esquema comum
    (COLUNAS), só com as linhas usáveis (benign/malignant, "normal" descartado)."""
    exigir(CLINICO.exists(), f"não encontrei a planilha clínica em {CLINICO}")
    bruto = pd.read_excel(CLINICO, dtype={"BIRADS": str})

    exigir(
        len(bruto) == LINHAS_BRUTAS_ESPERADAS,
        f"esperava {LINHAS_BRUTAS_ESPERADAS} linhas na planilha bruta, encontrei {len(bruto)}",
    )

    # --- rótulo bruto: só 3 valores possíveis, "normal" é descartado a seguir ----------
    exigir(bruto["Classification"].notna().all(), "há linha com Classification vazio na planilha")
    valores_aceitos = set(ROTULO_BREAST) | {"normal"}
    inesperados = set(bruto["Classification"].unique()) - valores_aceitos
    exigir(
        not inesperados,
        "valor inesperado em Classification: " + ", ".join(sorted(map(repr, inesperados))),
    )

    # --- descarta os "normal" ANTES de montar qualquer caminho de máscara --------------
    # Mask_tumor_filename vem NaN para essas linhas (não há lesão para segmentar); tentar
    # montar o caminho antes de descartar procuraria um arquivo que nunca existiu.
    normais = bruto["Classification"] == "normal"
    exigir(
        int(normais.sum()) == NORMAIS_ESPERADOS,
        f"esperava {NORMAIS_ESPERADOS} casos 'normal' para descartar, encontrei {int(normais.sum())}",
    )
    usaveis = bruto.loc[~normais].reset_index(drop=True)
    exigir(
        len(usaveis) == LINHAS_ESPERADAS,
        f"depois de descartar 'normal', esperava {LINHAS_ESPERADAS} linhas, encontrei {len(usaveis)}",
    )

    # Confirma que só benign/malignant sobraram (o mapeamento abaixo travaria de qualquer
    # forma com KeyError silencioso virando NaN, mas travar explicitamente aqui dá uma
    # mensagem melhor).
    inesperados_pos_filtro = set(usaveis["Classification"].unique()) - set(ROTULO_BREAST)
    exigir(
        not inesperados_pos_filtro,
        "sobrou valor inesperado em Classification depois de descartar 'normal': "
        + ", ".join(sorted(map(repr, inesperados_pos_filtro))),
    )

    # --- paciente: CaseID não pode se repetir ------------------------------------------
    exigir(
        usaveis["CaseID"].is_unique,
        "CaseID repetido na planilha do BrEaST — esperava 1 linha = 1 paciente",
    )

    # --- caminhos dos arquivos ----------------------------------------------------------
    exigir(usaveis["Image_filename"].notna().all(), "há Image_filename vazio (linha usável)")
    exigir(
        usaveis["Mask_tumor_filename"].notna().all(),
        "há Mask_tumor_filename vazio depois de descartar 'normal' — não deveria acontecer "
        "(só os 'normal' têm máscara vazia)",
    )

    imagens = usaveis["Image_filename"].map(lambda nome: (BREAST_IMAGENS / nome).as_posix())
    mascaras = usaveis["Mask_tumor_filename"].map(lambda nome: (BREAST_IMAGENS / nome).as_posix())

    faltando = [
        f"CaseID {cid} -> {relativo}"
        for cid, relativo in [*zip(usaveis["CaseID"], imagens), *zip(usaveis["CaseID"], mascaras)]
        if not (RAIZ_TCC / relativo).exists()
    ]
    exigir(
        not faltando,
        f"{len(faltando)} arquivo(s) da tabela não existem no disco; primeiros casos:\n  "
        + "\n  ".join(faltando[:10]),
    )

    # --- largura/altura: lidas da imagem real, planilha não declara --------------------
    tamanhos = [tamanho_da_imagem(RAIZ_TCC / caminho) for caminho in imagens]
    larguras = [t[0] for t in tamanhos]
    alturas = [t[1] for t in tamanhos]

    # --- sha1: mesmo cálculo de sempre (função importada de indexar.py) ----------------
    sha1s = [sha1_do_arquivo(RAIZ_TCC / caminho) for caminho in imagens]

    # id textual, legível, no mesmo espírito de "bus_0001-l" do BUS-BRA.
    ids = "breast_" + usaveis["CaseID"].astype(str).str.zfill(3)

    return pd.DataFrame(
        {
            "base": "breast",
            "id": ids,
            "paciente": usaveis["CaseID"],
            "imagem": imagens,
            "mascara": mascaras,
            "rotulo": usaveis["Classification"].map(ROTULO_BREAST),
            "birads": usaveis["BIRADS"],
            "aparelho": None,  # não existe coluna de aparelho/fabricante nesta planilha
            "largura": larguras,
            "altura": alturas,
            "lado": None,  # conceito do BUS-BRA, não se aplica ao BrEaST
            "sha1": sha1s,
        },
        columns=COLUNAS,
    )


def verificar_sanidade(tabela: pd.DataFrame) -> None:
    """Confere o índice montado antes de salvar — mesmo espírito de
    `indexar.verificar_sanidade`, mas SEM checar `aparelho`/`lado` como vazias: são
    vazias por decisão de projeto para o BrEaST (ver docstring do módulo), não por erro.
    """
    exigir(
        len(tabela) == LINHAS_ESPERADAS,
        f"esperava {LINHAS_ESPERADAS} linhas, encontrei {len(tabela)}",
    )

    pacientes = tabela["paciente"].nunique()
    exigir(
        pacientes == PACIENTES_ESPERADOS,
        f"esperava {PACIENTES_ESPERADOS} pacientes distintos, encontrei {pacientes}",
    )

    duplicados = tabela["sha1"].duplicated(keep=False)
    exigir(
        not duplicados.any(),
        f"{int(duplicados.sum())} imagem(ns) do BrEaST com sha1 repetido (conteúdo "
        "idêntico); primeiros casos:\n  "
        + "\n  ".join(tabela.loc[duplicados, "imagem"].head(10)),
    )

    vazias = [coluna for coluna in COLUNAS_CRITICAS if tabela[coluna].isna().any()]
    exigir(not vazias, "coluna(s) crítica(s) com valor vazio no índice: " + ", ".join(vazias))

    exigir(
        set(tabela["rotulo"].unique()) <= {0, 1},
        f"rótulo com valor inesperado depois do mapeamento: {sorted(tabela['rotulo'].unique())}",
    )


def numero(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def resumir(tabela: pd.DataFrame) -> None:
    malignas = int(tabela["rotulo"].sum())
    proporcao = f"{100 * malignas / len(tabela):.1f}".replace(".", ",")
    print(
        f"BREAST: {numero(len(tabela))} imagens, {numero(tabela['paciente'].nunique())} "
        f"pacientes, {numero(malignas)} malignas, {proporcao}%"
    )
    print("  aparelho: coluna vazia para todas as linhas (não existe na planilha do BrEaST)")
    print("  lado: coluna vazia para todas as linhas (conceito não se aplica ao BrEaST)")


def main() -> None:
    tabela = ler_breast()
    verificar_sanidade(tabela)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(SAIDA, index=False)

    print(f"índice do BrEaST salvo em {SAIDA.relative_to(RAIZ_TCC).as_posix()}")
    resumir(tabela)


if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
