"""Etapa 2 do pipeline: atribui cada imagem do indice a um dos 5 folds oficiais do BUS-BRA.

Le `dados_processados/indice.csv` (saida da Etapa 1) e o arquivo oficial de folds,
distribuido pelos proprios autores do dataset, `01-datasets/BUS-BRA/BUSBRA/5-fold-cv.csv`.
Junta as duas tabelas pela coluna `id` e escreve `dados_processados/indice_particionado.csv`.

Decisao ja registrada no projeto: usar os folds oficiais do BUS-BRA, nao criar uma
particao propria — da comparabilidade direta com o baseline publicado, e uma etapa de
implementacao a menos. O split "por imagem" de proposito (Protocolo B, o experimento de
vazamento) nao entra aqui: fica para o `experimento.py`.

A partir desta etapa, nenhum script seguinte (`verificar.py`, `features.py`,
`experimento.py`...) volta a abrir `5-fold-cv.csv`: todos leem so
`indice_particionado.csv`.

Uso:
    python src/comum/particionar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Caminhos: derivados da posicao deste arquivo (ver indexar.py)

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA_INDICE = RAIZ_CODIGO / "dados_processados" / "indice.csv"
ENTRADA_FOLDS = RAIZ_TCC / "01-datasets" / "BUS-BRA" / "BUSBRA" / "5-fold-cv.csv"

SAIDA = RAIZ_CODIGO / "dados_processados" / "indice_particionado.csv"

# Numeros conferidos antes deste script (docs/02-particionar.md, secao 5), para sanidade
LINHAS_ESPERADAS = 1875
CONTAGEM_POR_FOLD_ESPERADA = {1: 376, 2: 385, 3: 366, 4: 365, 5: 383}


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com mensagem clara se a condicao nao valer."""
    if not condicao:
        raise ValueError(mensagem)


def ler_indice() -> pd.DataFrame:
    """Le o indice da Etapa 1; `birads` como texto para nao perder formatos como "4a"."""
    exigir(ENTRADA_INDICE.exists(), f"não encontrei {ENTRADA_INDICE}; rode indexar.py antes")
    return pd.read_csv(ENTRADA_INDICE, dtype={"birads": str})


def ler_folds_oficiais() -> pd.DataFrame:
    """Le o arquivo oficial de folds, mantendo so `ID` e `kFold` (valid_1..valid_5 descartadas)."""
    exigir(ENTRADA_FOLDS.exists(), f"não encontrei {ENTRADA_FOLDS}")
    bruto = pd.read_csv(ENTRADA_FOLDS)
    exigir(
        {"ID", "kFold"}.issubset(bruto.columns),
        f"esperava as colunas ID e kFold em {ENTRADA_FOLDS.name}, encontrei {list(bruto.columns)}",
    )
    return bruto[["ID", "kFold"]]


def juntar(indice: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Junta indice e folds oficiais pelo `id` e renomeia `kFold` para `fold`."""
    juntado = indice.merge(folds, left_on="id", right_on="ID", how="left")
    juntado = juntado.drop(columns=["ID"]).rename(columns={"kFold": "fold"})

    # A juncao nao pode perder nem duplicar linhas do indice.
    exigir(
        len(juntado) == len(indice),
        f"a junção mudou o número de linhas: {len(indice)} -> {len(juntado)} "
        "(há id duplicado em algum dos dois arquivos)",
    )

    # Todo id do indice precisa ter fold no arquivo oficial.
    sem_fold = juntado[juntado["fold"].isna()]
    exigir(
        sem_fold.empty,
        f"{len(sem_fold)} imagem(ns) do índice não encontraram fold em "
        f"{ENTRADA_FOLDS.name}; primeiros ids:\n  "
        + "\n  ".join(sem_fold["id"].astype(str).head(10)),
    )

    juntado["fold"] = juntado["fold"].astype(int)
    return juntado


def verificar_gate_por_paciente(tabela: pd.DataFrame) -> None:
    """Gate central: todas as imagens de um paciente devem cair no mesmo fold (sem vazamento)."""
    folds_por_paciente = tabela.groupby("paciente")["fold"].nunique()
    pacientes_com_mais_de_um_fold = folds_por_paciente[folds_por_paciente > 1]
    exigir(
        pacientes_com_mais_de_um_fold.empty,
        f"{len(pacientes_com_mais_de_um_fold)} paciente(s) com imagens em mais de um "
        "fold (vazamento entre treino e teste); primeiros casos:\n  "
        + "\n  ".join(str(p) for p in pacientes_com_mais_de_um_fold.head(10).index)
    )


def verificar_sanidade(tabela: pd.DataFrame) -> None:
    """Confere o resultado contra os numeros esperados (docs/02-particionar.md, secao 5)."""
    exigir(
        len(tabela) == LINHAS_ESPERADAS,
        f"esperava {LINHAS_ESPERADAS} linhas, encontrei {len(tabela)}",
    )

    contagem = tabela["fold"].value_counts().to_dict()
    exigir(
        contagem == CONTAGEM_POR_FOLD_ESPERADA,
        f"contagem por fold não bate com o esperado; esperava "
        f"{CONTAGEM_POR_FOLD_ESPERADA}, encontrei {contagem}",
    )

    verificar_gate_por_paciente(tabela)


def numero(valor: int) -> str:
    """Formata um inteiro com ponto de milhar (1875 -> '1.875')."""
    return f"{valor:,}".replace(",", ".")


def resumir(tabela: pd.DataFrame) -> None:
    """Imprime o resumo usado em Materiais e Metodos."""
    print(f"{numero(len(tabela))} imagens em 5 folds:")
    for fold, quantidade in sorted(tabela["fold"].value_counts().items()):
        print(f"  fold {fold}: {numero(int(quantidade))} imagens")


def main() -> None:
    indice = ler_indice()
    folds = ler_folds_oficiais()

    juntado = juntar(indice, folds)
    verificar_sanidade(juntado)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    juntado.to_csv(SAIDA, index=False)

    print(f"indice particionado salvo em {SAIDA.relative_to(RAIZ_TCC).as_posix()}")
    resumir(juntado)


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
