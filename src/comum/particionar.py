"""Etapa 2 do pipeline: atribui cada imagem do índice a um dos 5 folds oficiais do BUS-BRA.

Lê `dados_processados/indice.csv` (saída da Etapa 1) e o arquivo oficial de folds,
distribuído pelos próprios autores do dataset, `01-datasets/BUS-BRA/BUSBRA/5-fold-cv.csv`.
Junta as duas tabelas pela coluna `id` e escreve `dados_processados/indice_particionado.csv`.

Decisão já registrada no projeto: usar os folds oficiais do BUS-BRA, não criar uma
partição própria — dá comparabilidade direta com o baseline publicado, e uma etapa de
implementação a menos. O split "por imagem" de propósito (Protocolo B, o experimento de
vazamento) não entra aqui: fica para o `experimento.py`.

A partir desta etapa, nenhum script seguinte (`verificar.py`, `features.py`,
`experimento.py`...) volta a abrir `5-fold-cv.csv`: todos leem só
`indice_particionado.csv`.

Uso:
    python src/comum/particionar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos — tudo derivado da posição deste arquivo (ver indexar.py).
# ---------------------------------------------------------------------------

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA_INDICE = RAIZ_CODIGO / "dados_processados" / "indice.csv"
ENTRADA_FOLDS = RAIZ_TCC / "01-datasets" / "BUS-BRA" / "BUSBRA" / "5-fold-cv.csv"

SAIDA = RAIZ_CODIGO / "dados_processados" / "indice_particionado.csv"

# Números conferidos por conferência independente antes deste script existir (ver
# docs/02-particionar.md, seção 5) — usados como verificação de sanidade.
LINHAS_ESPERADAS = 1875
CONTAGEM_POR_FOLD_ESPERADA = {1: 376, 2: 385, 3: 366, 4: 365, 5: 383}


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com uma mensagem clara se a condição não valer.

    Mesma função de `indexar.py`: a regra é parar com erro, nunca seguir em frente com
    dado suspeito.
    """
    if not condicao:
        raise ValueError(mensagem)


def ler_indice() -> pd.DataFrame:
    """Lê o índice da Etapa 1. A coluna `birads` precisa ser lida como texto — se não for
    especificada aqui, o pandas infere número e perde o formato (`"4a"` no BrEaST, mais
    para frente)."""
    exigir(ENTRADA_INDICE.exists(), f"não encontrei {ENTRADA_INDICE}; rode indexar.py antes")
    return pd.read_csv(ENTRADA_INDICE, dtype={"birads": str})


def ler_folds_oficiais() -> pd.DataFrame:
    """Lê o arquivo oficial de folds, mantendo só `ID` e `kFold`.

    As colunas `valid_1`...`valid_5` do arquivo original parecem ser máscaras auxiliares
    dos autores para outro propósito (validação cruzada aninhada, possivelmente) — não
    são necessárias para o protocolo deste TCC e são descartadas aqui, de propósito.
    """
    exigir(ENTRADA_FOLDS.exists(), f"não encontrei {ENTRADA_FOLDS}")
    bruto = pd.read_csv(ENTRADA_FOLDS)
    exigir(
        {"ID", "kFold"}.issubset(bruto.columns),
        f"esperava as colunas ID e kFold em {ENTRADA_FOLDS.name}, encontrei {list(bruto.columns)}",
    )
    return bruto[["ID", "kFold"]]


def juntar(indice: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Junta o índice com os folds oficiais pela coluna `id`/`ID` e renomeia `kFold` para
    `fold`."""
    juntado = indice.merge(folds, left_on="id", right_on="ID", how="left")
    juntado = juntado.drop(columns=["ID"]).rename(columns={"kFold": "fold"})

    # A junção não pode ter perdido nem duplicado nenhuma linha do índice original.
    exigir(
        len(juntado) == len(indice),
        f"a junção mudou o número de linhas: {len(indice)} -> {len(juntado)} "
        "(há id duplicado em algum dos dois arquivos)",
    )

    # Nenhum id do índice pode ter ficado sem fold correspondente no arquivo oficial.
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
    """O "gate" central desta etapa: todas as imagens de um mesmo paciente têm que cair
    no mesmo fold.

    Se um paciente tiver imagens espalhadas em folds diferentes, treinar num fold e
    testar noutro vazaria informação do mesmo paciente entre treino e teste — o próprio
    tipo de vazamento que o Protocolo B (proposital, em experimento.py) existe para
    medir. Aqui, no particionamento oficial, isso não pode acontecer OU o arquivo oficial
    de folds já parte do princípio de não misturar pacientes entre folds — auditar essa
    suposição, em vez de confiar cegamente nela, é o argumento de rigor metodológico
    central deste TCC.
    """
    folds_por_paciente = tabela.groupby("paciente")["fold"].nunique()
    pacientes_com_mais_de_um_fold = folds_por_paciente[folds_por_paciente > 1]
    exigir(
        pacientes_com_mais_de_um_fold.empty,
        f"{len(pacientes_com_mais_de_um_fold)} paciente(s) com imagens em mais de um "
        "fold (vazamento entre treino e teste); primeiros casos:\n  "
        + "\n  ".join(str(p) for p in pacientes_com_mais_de_um_fold.head(10).index)
    )


def verificar_sanidade(tabela: pd.DataFrame) -> None:
    """Confere o resultado contra os números já conferidos antes de salvar (ver
    docs/02-particionar.md, seção 5)."""
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
    """Imprime o resumo que entra na seção de Materiais e Métodos."""
    print(f"{numero(len(tabela))} imagens particionadas em 5 folds oficiais:")
    for fold, quantidade in sorted(tabela["fold"].value_counts().items()):
        print(f"  fold {fold}: {numero(int(quantidade))} imagens")
    print("0 pacientes com imagens em mais de um fold (gate verificado).")


def main() -> None:
    indice = ler_indice()
    folds = ler_folds_oficiais()

    juntado = juntar(indice, folds)
    verificar_sanidade(juntado)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    juntado.to_csv(SAIDA, index=False)

    print(f"índice particionado salvo em {SAIDA.relative_to(RAIZ_TCC).as_posix()}")
    resumir(juntado)


if __name__ == "__main__":
    # O console do Windows costuma abrir em cp1252 e comeria os acentos das mensagens.
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
