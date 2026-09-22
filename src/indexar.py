"""Etapa 1 do pipeline: monta o índice único do projeto a partir dos dados brutos.

Lê apenas os arquivos brutos do BUS-BRA (`bus_data.csv`, `Images/`, `Masks/`) e escreve
um único arquivo, `dados_processados/indice.csv`, no esquema comum descrito abaixo.

A partir daqui, nenhum outro script do projeto (`particionar.py`, `features.py`,
`experimento.py`...) volta a abrir os dados brutos: todos leem só o `indice.csv`. Assim,
se um número aparecer errado depois, ou o erro está aqui (montagem do índice) ou está lá
na frente — nunca nos dois lugares ao mesmo tempo.

Uso:
    python src/indexar.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos
#
# Tudo é derivado da posição deste arquivo, nunca de um caminho absoluto escrito
# na mão: o script continua funcionando se a pasta for movida ou rodar em outra
# máquina.
# ---------------------------------------------------------------------------

RAIZ_CODIGO = Path(__file__).resolve().parents[1]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

# Os caminhos gravados no índice são relativos a RAIZ_TCC (a pasta que contém tanto
# `01-datasets/` quanto `03-codigo/`). Quem consumir o índice depois deve resolvê-los
# contra essa mesma raiz.
BUSBRA = Path("01-datasets/BUS-BRA/BUSBRA")

SAIDA = RAIZ_CODIGO / "dados_processados" / "indice.csv"

# ---------------------------------------------------------------------------
# Esquema comum de saída (o BrEaST, quando entrar, devolve exatamente estas colunas)
# ---------------------------------------------------------------------------

COLUNAS = [
    "base",       # texto  — "bus-bra"; "breast" quando o segundo conjunto entrar
    "id",         # texto  — identificador da imagem original (ex.: "bus_0001-l"); usado para
                  # juntar com o arquivo oficial de folds na Etapa 2 (particionar.py)
    "paciente",   # número — identificador usado no split por paciente (Etapa 2)
    "imagem",     # texto  — caminho do PNG, relativo a RAIZ_TCC
    "mascara",    # texto  — caminho da máscara, relativo a RAIZ_TCC
    "rotulo",     # 0 ou 1 — 1 = malignant, 0 = benign
    "birads",     # texto  — texto de propósito: no BrEaST vem "4a", "4b", ...
    "aparelho",   # texto  — para a análise por subgrupo, mais para frente
    "largura",    # número
    "altura",     # número
    "lado",       # texto  — left / right / single; só carrega a informação adiante
    "sha1",       # texto  — impressão digital do conteúdo do PNG
]

# Tradução do rótulo. O dicionário também serve de lista fechada de valores aceitos:
# qualquer outro valor na coluna `Pathology` faz o script travar.
ROTULO_BUSBRA = {"benign": 0, "malignant": 1}

# Números conferidos no conjunto bruto, usados como verificação de sanidade.
LINHAS_ESPERADAS = 1875
PACIENTES_ESPERADOS = 1064


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com uma mensagem clara se a condição não valer.

    Toda verificação do script passa por aqui: a regra é parar com erro, nunca pular
    a linha problemática em silêncio e seguir em frente.
    """
    if not condicao:
        raise ValueError(mensagem)


def sha1_do_arquivo(caminho: Path) -> str:
    """Impressão digital do conteúdo do arquivo.

    Não é sobre segurança: dois arquivos com o mesmo hash são byte a byte idênticos,
    o que permite achar imagens duplicadas depois, de graça.
    """
    return hashlib.sha1(caminho.read_bytes()).hexdigest()


def ler_bus_bra() -> pd.DataFrame:
    """Lê o BUS-BRA bruto e devolve a tabela já no esquema comum (COLUNAS).

    Esta função é o modelo para a futura `ler_breast()`: toda particularidade do
    conjunto (nome das colunas, convenção dos nomes de arquivo, valores do rótulo) fica
    contida aqui; o resto do script não sabe de que conjunto veio a tabela.
    """
    caminho_csv = RAIZ_TCC / BUSBRA / "bus_data.csv"
    exigir(caminho_csv.exists(), f"não encontrei o CSV bruto em {caminho_csv}")

    # BIRADS entra como texto de propósito (ver COLUNAS).
    bruto = pd.read_csv(caminho_csv, dtype={"BIRADS": str})

    # --- caminhos dos arquivos ---------------------------------------------------
    # As imagens seguem o ID direto:         bus_0001-l -> Images/bus_0001-l.png
    # As máscaras NÃO seguem o mesmo padrão: o prefixo `bus_` vira `mask_`
    #                                        bus_0001-l -> Masks/mask_0001-l.png
    # Reusar o ID para montar o nome da máscara procuraria um arquivo inexistente;
    # pior, uma convenção errada aqui associaria em silêncio a máscara de um paciente
    # à imagem de outro.
    ids = bruto["ID"].astype(str)
    fora_do_padrao = ids[~ids.str.startswith("bus_")]
    exigir(
        fora_do_padrao.empty,
        "ID fora do padrão esperado (deveria começar com 'bus_'): "
        + ", ".join(fora_do_padrao.head(5)),
    )

    imagens = ids.map(lambda i: (BUSBRA / "Images" / f"{i}.png").as_posix())
    mascaras = ids.map(
        lambda i: (BUSBRA / "Masks" / f"mask_{i.removeprefix('bus_')}.png").as_posix()
    )

    # --- os arquivos existem mesmo no disco? -------------------------------------
    # Parece exagero para 1.875 imagens que já sabemos que batem, mas é o mesmo
    # princípio do `verificar.py` da Etapa 3 — e o TCC inteiro é sobre como esse tipo
    # de verificação costuma faltar na área.
    faltando = [
        f"{identificador} -> {relativo}"
        for identificador, relativo in [*zip(ids, imagens), *zip(ids, mascaras)]
        if not (RAIZ_TCC / relativo).exists()
    ]
    exigir(
        not faltando,
        f"{len(faltando)} arquivo(s) da tabela não existem no disco; primeiros casos:\n  "
        + "\n  ".join(faltando[:10]),
    )

    # --- rótulo -------------------------------------------------------------------
    # Além de traduzir, conferir que só os dois valores conhecidos aparecem. Parece
    # redundante aqui, mas é essa verificação que vai obrigar a decidir explicitamente
    # o que fazer com o terceiro valor do BrEaST (`normal`), em vez de deixá-lo vazar
    # para dentro do treino sem querer.
    exigir(
        bruto["Pathology"].notna().all(),
        "há linhas com Pathology vazio em bus_data.csv",
    )
    inesperados = set(bruto["Pathology"].unique()) - set(ROTULO_BUSBRA)
    exigir(
        not inesperados,
        "valor inesperado em Pathology: " + ", ".join(sorted(map(repr, inesperados))),
    )

    return pd.DataFrame(
        {
            "base": "bus-bra",
            "id": ids,
            "paciente": bruto["Case"],
            "imagem": imagens,
            "mascara": mascaras,
            "rotulo": bruto["Pathology"].map(ROTULO_BUSBRA),
            "birads": bruto["BIRADS"],
            "aparelho": bruto["Device"],
            "largura": bruto["Width"],
            "altura": bruto["Height"],
            "lado": bruto["Side"],
            "sha1": [sha1_do_arquivo(RAIZ_TCC / caminho) for caminho in imagens],
        },
        columns=COLUNAS,
    )


def verificar_sanidade(tabela: pd.DataFrame) -> None:
    """Confere o índice montado antes de salvar qualquer coisa.

    Se algo aqui falhar, nenhum arquivo é escrito: melhor não ter índice do que ter um
    índice quebrado passando por bom.
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
        f"{int(duplicados.sum())} imagem(ns) com sha1 repetido (conteúdo idêntico); "
        "primeiros casos:\n  "
        + "\n  ".join(tabela.loc[duplicados, "imagem"].head(10)),
    )

    vazias = [coluna for coluna in COLUNAS if tabela[coluna].isna().any()]
    exigir(not vazias, "coluna(s) com valor vazio no índice: " + ", ".join(vazias))


def numero(valor: int) -> str:
    """Formata um inteiro com ponto de milhar (1875 -> '1.875')."""
    return f"{valor:,}".replace(",", ".")


def resumir(tabela: pd.DataFrame) -> None:
    """Imprime as contagens que entram na seção de Materiais e Métodos."""
    for base, parte in tabela.groupby("base", sort=True):
        malignas = int(parte["rotulo"].sum())
        proporcao = f"{100 * malignas / len(parte):.1f}".replace(".", ",")
        print(
            f"{base.upper()}: {numero(len(parte))} imagens, "
            f"{numero(parte['paciente'].nunique())} pacientes, "
            f"{numero(malignas)} malignas, {proporcao}%"
        )


def main() -> None:
    tabela = ler_bus_bra()
    verificar_sanidade(tabela)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(SAIDA, index=False)

    print(f"índice salvo em {SAIDA.relative_to(RAIZ_TCC).as_posix()}")
    resumir(tabela)


if __name__ == "__main__":
    # O console do Windows costuma abrir em cp1252 e comeria os acentos das mensagens.
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
