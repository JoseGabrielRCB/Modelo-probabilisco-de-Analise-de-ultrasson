"""Etapa 1 (BrEaST) do Protocolo C: monta o indice do BrEaST no mesmo esquema do BUS-BRA,
para uso so como validacao externa (nunca como treino).

Le a planilha clinica `BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx` e a pasta de
imagens/mascaras, e escreve `dados_processados/breast/indice_breast.csv` com as mesmas
`COLUNAS` de `comum/indexar.py` (importadas, nao copiadas).

Particularidades do BrEaST:
    - 256 linhas, das quais 4 sao "normal" (sem lesao, sem mascara): descartadas antes de
      montar qualquer caminho de mascara. Sobram 252, benignas ou malignas.
    - `paciente` = `CaseID` (1 imagem por paciente).
    - `birads` mantido como texto ("4a", "4b", ...).
    - Nao ha aparelho nem lado na planilha: `aparelho` e `lado` ficam vazios de proposito.
    - `largura`/`altura` lidas da imagem real (a planilha nao as declara).

Uso:
    python src/protocolo_c/indexar_breast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from comum.indexar import COLUNAS, exigir, numero, sha1_do_arquivo

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

BREAST = Path("01-datasets/BrEaST-Lesions-USG")
BREAST_IMAGENS = BREAST / "BrEaST-Lesions_USG-images_and_masks"
CLINICO = RAIZ_TCC / BREAST / "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"

SAIDA = RAIZ_CODIGO / "dados_processados" / "breast" / "indice_breast.csv"

# Traducao do rotulo; tambem e a lista fechada de valores aceitos (alem de "normal").
ROTULO_BREAST = {"benign": 0, "malignant": 1}

# Numeros conferidos no conjunto bruto (256 linhas na planilha, 4 "normal" descartadas).
LINHAS_BRUTAS_ESPERADAS = 256
NORMAIS_ESPERADOS = 4
LINHAS_ESPERADAS = 252
PACIENTES_ESPERADOS = 252

# `aparelho` e `lado` ficam de fora: sao vazias por decisao de projeto.
COLUNAS_CRITICAS = ["base", "id", "paciente", "imagem", "mascara", "rotulo", "birads", "sha1"]


def tamanho_da_imagem(caminho: Path) -> tuple[int, int]:
    """(largura, altura) lidos do arquivo real."""
    with Image.open(caminho) as img:
        return img.size


def ler_breast() -> pd.DataFrame:
    """Le a planilha bruta e devolve as 252 linhas usaveis no esquema comum (COLUNAS)."""
    exigir(CLINICO.exists(), f"não encontrei a planilha clínica em {CLINICO}")
    bruto = pd.read_excel(CLINICO, dtype={"BIRADS": str})

    exigir(
        len(bruto) == LINHAS_BRUTAS_ESPERADAS,
        f"esperava {LINHAS_BRUTAS_ESPERADAS} linhas na planilha bruta, encontrei {len(bruto)}",
    )

    # Rotulo bruto: so benign, malignant ou normal
    exigir(bruto["Classification"].notna().all(), "há linha com Classification vazio na planilha")
    inesperados = set(bruto["Classification"].unique()) - (set(ROTULO_BREAST) | {"normal"})
    exigir(
        not inesperados,
        "valor inesperado em Classification: " + ", ".join(sorted(map(repr, inesperados))),
    )

    # Descarta os "normal" ANTES de montar caminhos de mascara (eles nao tem mascara)
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

    exigir(
        usaveis["CaseID"].is_unique,
        "CaseID repetido na planilha do BrEaST — esperava 1 linha = 1 paciente",
    )

    # Caminhos dos arquivos
    exigir(usaveis["Image_filename"].notna().all(), "há Image_filename vazio (linha usável)")
    exigir(
        usaveis["Mask_tumor_filename"].notna().all(),
        "há Mask_tumor_filename vazio depois de descartar 'normal'",
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

    tamanhos = [tamanho_da_imagem(RAIZ_TCC / caminho) for caminho in imagens]

    return pd.DataFrame(
        {
            "base": "breast",
            "id": "breast_" + usaveis["CaseID"].astype(str).str.zfill(3),
            "paciente": usaveis["CaseID"],
            "imagem": imagens,
            "mascara": mascaras,
            "rotulo": usaveis["Classification"].map(ROTULO_BREAST),
            "birads": usaveis["BIRADS"],
            "aparelho": None,  # nao existe na planilha do BrEaST
            "largura": [t[0] for t in tamanhos],
            "altura": [t[1] for t in tamanhos],
            "lado": None,  # conceito do BUS-BRA, nao se aplica ao BrEaST
            "sha1": [sha1_do_arquivo(RAIZ_TCC / caminho) for caminho in imagens],
        },
        columns=COLUNAS,
    )


def verificar_sanidade(tabela: pd.DataFrame) -> None:
    """Confere o indice antes de salvar (sem exigir `aparelho`/`lado`, vazias de proposito)."""
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


def resumir(tabela: pd.DataFrame) -> None:
    malignas = int(tabela["rotulo"].sum())
    proporcao = f"{100 * malignas / len(tabela):.1f}".replace(".", ",")
    print(
        f"BREAST: {numero(len(tabela))} imagens, {numero(tabela['paciente'].nunique())} "
        f"pacientes, {numero(malignas)} malignas, {proporcao}%"
    )


def main() -> None:
    tabela = ler_breast()
    verificar_sanidade(tabela)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(SAIDA, index=False)

    print(f"indice do BrEaST salvo em {SAIDA.relative_to(RAIZ_TCC).as_posix()}")
    resumir(tabela)


if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        # Usar esse metodo é um pouco estranho , mas foi nescessario para poder validar um erro ocorrido e tambem por questao de docuemntao
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
