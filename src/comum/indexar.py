"""Etapa 1 do pipeline: monta o indice unico do projeto a partir dos dados brutos.

Le apenas os arquivos brutos do BUS-BRA (`bus_data.csv`, `Images/`, `Masks/`) e escreve
um unico arquivo, `dados_processados/indice.csv`, no esquema comum descrito abaixo.

A partir daqui, nenhum outro script do projeto (`particionar.py`, `features.py`,
`experimento.py`...) volta a abrir os dados brutos: todos leem so o `indice.csv`. Assim,
se um numero aparecer errado depois, ou o erro esta aqui (montagem do indice) ou esta la
na frente — nunca nos dois lugares ao mesmo tempo.

Uso:
    python src/comum/indexar.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


# Caminhos
RAIZ_CODIGO = Path(__file__).resolve().parents[2]
RAIZ_TCC = RAIZ_CODIGO.parent


BUSBRA = Path("01-datasets/BUS-BRA/BUSBRA")

SAIDA = RAIZ_CODIGO / "dados_processados" / "indice.csv"

# Esquema comum de saida (o BrEaST usa as mesmas colunas)

COLUNAS = [
    "base",       # texto: "bus-bra" ou "breast"
    "id",         # texto: id da imagem original (ex.: "bus_0001-l"), junta com os folds
    "paciente",   # numero: id usado no split por paciente (Etapa 2)
    "imagem",     # texto: caminho do PNG, relativo a RAIZ_TCC
    "mascara",    # texto: caminho da mascara, relativo a RAIZ_TCC
    "rotulo",     # 0 ou 1: 1 = malignant, 0 = benign
    "birads",     # texto de proposito: no BrEaST vem "4a", "4b", ...
    "aparelho",   # texto: para analise por subgrupo
    "largura",    # numero
    "altura",     # numero
    "lado",       # texto: left / right / single
    "sha1",       # texto: impressao digital do conteudo do PNG
]

# Traducao do rotulo; qualquer outro valor em `Pathology` trava o script
ROTULO_BUSBRA = {"benign": 0, "malignant": 1}

# Numeros conferidos no conjunto bruto (verificacao de sanidade)
LINHAS_ESPERADAS = 1875
PACIENTES_ESPERADOS = 1064


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com mensagem clara se a condicao nao valer."""
    if not condicao:
        raise ValueError(mensagem)


def sha1_do_arquivo(caminho: Path) -> str:
    """Hash do conteudo do arquivo; hashes iguais indicam imagens duplicadas."""
    return hashlib.sha1(caminho.read_bytes()).hexdigest()


def ler_bus_bra() -> pd.DataFrame:
    """Le o BUS-BRA bruto e devolve a tabela no esquema comum (COLUNAS)."""
    caminho_csv = RAIZ_TCC / BUSBRA / "bus_data.csv"
    exigir(caminho_csv.exists(), f"não encontrei o CSV bruto em {caminho_csv}")

    # BIRADS entra como texto de proposito (ver COLUNAS).
    bruto = pd.read_csv(caminho_csv, dtype={"BIRADS": str})

    # Imagem: bus_0001-l -> Images/bus_0001-l.png; mascara: bus_0001-l -> Masks/mask_0001-l.png
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

    # Confere se os arquivos existem no disco (mesmo principio do verificar.py)
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

    # Rotulo: so aceita os dois valores conhecidos (o BrEaST tem um terceiro, `normal`)
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
    """Confere o indice antes de salvar; se falhar, nada e escrito."""
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
        "primeiros casos:\n  "+ "\n  ".join(tabela.loc[duplicados, "imagem"].head(10)),
    )

    vazias = [coluna for coluna in COLUNAS if tabela[coluna].isna().any()]
    exigir(not vazias, "coluna(s) com valor vazio no índice: " + ", ".join(vazias))


def numero(valor: int) -> str:
    """Formata um inteiro com ponto de milhar (1875 -> '1.875')."""
    return f"{valor:,}".replace(",", ".")


def resumir(tabela: pd.DataFrame) -> None:
    """Imprime as contagens usadas em Materiais e Metodos."""
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

    print(f"indice salvo em {SAIDA.relative_to(RAIZ_TCC).as_posix()}")
    resumir(tabela)


if __name__ == "__main__":
    # Console do Windows abre em cp1252; forca UTF-8 para nao perder acentos
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        # Usar esse metodo é um pouco estranho , mas foi nescessario para poder validar um erro ocorrido e tambem por questao de docuemntao
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
