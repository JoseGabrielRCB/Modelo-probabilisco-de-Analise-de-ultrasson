"""Etapa 5 do pipeline: treina o classificador e coleta predições fora-de-amostra.

Lê `dados_processados/caracteristicas.npy` e `dados_processados/caracteristicas_ids.csv`
(saída de `features.py`) — nunca lê `indice_particionado.csv` diretamente, para que
qualquer linha usada aqui já tenha passado pelo `verificar.py` antes dela existir como
característica.

Classificador: `Pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
class_weight="balanced"))`. `class_weight="balanced"` compensa o desbalanço de classes
(32% malignas) sem precisar reamostrar os dados manualmente. Semente fixa 42 em tudo que
aceita `random_state`, para que rodar o script duas vezes produza exatamente os mesmos
números.

Dois protocolos, o mesmo classificador nos dois:

    Protocolo A (principal) — validação cruzada usando os 5 folds OFICIAIS já presentes
    na coluna `fold` (agrupados por paciente, conferidos em `particionar.py`/
    `verificar.py`). Para cada fold como teste, treina nos outros 4 e prediz no fold de
    teste. Ao final, toda imagem tem uma predição fora-de-amostra. Depois, agrega por
    paciente (probabilidade média das imagens do mesmo paciente) e grava essas linhas
    também, com `unidade=paciente`.

    Protocolo B (vazamento, só para comparação) — os MESMOS dados, mas particionados com
    `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, que ignora de propósito
    o agrupamento por paciente. Como a mesma pessoa pode ter imagens (esquerda/direita)
    em treino E em teste ao mesmo tempo, o modelo pode "decorar" características do
    paciente em vez de aprender a distinguir benigno de maligno — o Protocolo B existe só
    para medir o tamanho desse efeito (Etapa 6 compara A vs B), não para ser usado como
    resultado principal do TCC.

Escreve `resultados/predicoes.csv`, com uma linha por (protocolo, unidade, observação):
    protocolo (A ou B), unidade (imagem ou paciente), dataset (bus-bra), id, paciente,
    fold, y_true, y_score.

Uso:
    python src/experimento.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Caminhos — tudo derivado da posição deste arquivo (ver indexar.py).
# ---------------------------------------------------------------------------

RAIZ_CODIGO = Path(__file__).resolve().parents[1]   # .../TCC-Ultrassom/03-codigo

ENTRADA_MATRIZ = RAIZ_CODIGO / "dados_processados" / "caracteristicas.npy"
ENTRADA_IDS = RAIZ_CODIGO / "dados_processados" / "caracteristicas_ids.csv"

SAIDA = RAIZ_CODIGO / "resultados" / "predicoes.csv"

SEMENTE = 42
N_FOLDS_PROTOCOLO_B = 5
COLUNAS_SAIDA = ["protocolo", "unidade", "dataset", "id", "paciente", "fold", "y_true", "y_score"]


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com uma mensagem clara se a condição não valer."""
    if not condicao:
        raise ValueError(mensagem)


def ler_caracteristicas() -> tuple[np.ndarray, pd.DataFrame]:
    exigir(ENTRADA_MATRIZ.exists(), f"não encontrei {ENTRADA_MATRIZ}; rode features.py antes")
    exigir(ENTRADA_IDS.exists(), f"não encontrei {ENTRADA_IDS}; rode features.py antes")

    matriz = np.load(ENTRADA_MATRIZ)
    ids_tabela = pd.read_csv(ENTRADA_IDS)

    exigir(
        matriz.shape[0] == len(ids_tabela),
        f"matriz tem {matriz.shape[0]} linhas, tabela de ids tem {len(ids_tabela)} — "
        "features.py precisa ser rodado de novo (arquivos dessincronizados)",
    )
    exigir(
        set(ids_tabela["rotulo"].unique()) <= {0, 1},
        f"valor inesperado em rotulo: {sorted(ids_tabela['rotulo'].unique())}",
    )
    return matriz, ids_tabela


def novo_pipeline() -> Pipeline:
    """Constrói um Pipeline novo a cada chamada — nunca reaproveitar um pipeline já
    treinado entre folds, ou o StandardScaler do fold anterior vazaria estatísticas do
    fold seguinte."""
    return Pipeline([
        ("escala", StandardScaler()),
        ("logistica", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEMENTE,
        )),
    ])


def rodar_protocolo_a(matriz: np.ndarray, ids_tabela: pd.DataFrame) -> pd.DataFrame:
    """Validação cruzada pelos 5 folds oficiais (coluna `fold`, agrupados por paciente).
    Devolve as linhas de nível imagem E as agregadas por paciente."""
    y = ids_tabela["rotulo"].to_numpy()
    fold = ids_tabela["fold"].to_numpy()

    y_score = np.full(len(y), np.nan)
    for f in sorted(np.unique(fold)):
        treino = fold != f
        teste = fold == f
        pipeline = novo_pipeline()
        pipeline.fit(matriz[treino], y[treino])
        y_score[teste] = pipeline.predict_proba(matriz[teste])[:, 1]

    exigir(not np.isnan(y_score).any(), "sobrou predição vazia no Protocolo A")

    linhas_imagem = pd.DataFrame({
        "protocolo": "A",
        "unidade": "imagem",
        "dataset": "bus-bra",
        "id": ids_tabela["id"],
        "paciente": ids_tabela["paciente"],
        "fold": fold,
        "y_true": y,
        "y_score": y_score,
    })

    # --- agregação por paciente: probabilidade média das imagens do mesmo paciente ---
    # `fold` e `rotulo` são garantidamente uniformes dentro de um mesmo paciente (é
    # exatamente o que o "gate" de particionar.py/verificar.py garante para fold, e o
    # rotulo já vinha uniforme do próprio bus_data.csv — conferido antes de assumir isso
    # aqui: ver checagem logo abaixo).
    agregado = linhas_imagem.groupby("paciente", as_index=False).agg(
        y_true_media=("y_true", "mean"),
        y_score=("y_score", "mean"),
        fold=("fold", "first"),
        n_valores_distintos_de_fold=("fold", "nunique"),
        n_valores_distintos_de_rotulo=("y_true", "nunique"),
    )
    exigir(
        (agregado["n_valores_distintos_de_fold"] == 1).all(),
        "paciente com imagens em mais de um fold escapou do gate de particionar.py — "
        "não deveria ser possível chegar até aqui",
    )
    exigir(
        (agregado["n_valores_distintos_de_rotulo"] == 1).all(),
        "paciente com rótulo (benigno/maligno) não uniforme entre as próprias imagens — "
        "inesperado, dado que bus_data.csv já garantia isso por paciente",
    )

    linhas_paciente = pd.DataFrame({
        "protocolo": "A",
        "unidade": "paciente",
        "dataset": "bus-bra",
        "id": agregado["paciente"].astype(str),  # id repete o valor de paciente, ver spec
        "paciente": agregado["paciente"],
        "fold": agregado["fold"],
        "y_true": agregado["y_true_media"].round().astype(int),
        "y_score": agregado["y_score"],
    })

    return pd.concat([linhas_imagem, linhas_paciente], ignore_index=True)[COLUNAS_SAIDA]


def rodar_protocolo_b(matriz: np.ndarray, ids_tabela: pd.DataFrame) -> pd.DataFrame:
    """Os mesmos dados do Protocolo A, mas particionados com StratifiedKFold aleatório
    por IMAGEM, ignorando de propósito o agrupamento por paciente — existe só para medir
    o efeito do vazamento (comparação feita em metricas.py), não é um resultado a favor
    do modelo."""
    y = ids_tabela["rotulo"].to_numpy()

    fold_b = np.zeros(len(y), dtype=int)
    y_score = np.full(len(y), np.nan)

    divisor = StratifiedKFold(n_splits=N_FOLDS_PROTOCOLO_B, shuffle=True, random_state=SEMENTE)
    for numero_do_fold, (treino, teste) in enumerate(divisor.split(matriz, y), start=1):
        fold_b[teste] = numero_do_fold
        pipeline = novo_pipeline()
        pipeline.fit(matriz[treino], y[treino])
        y_score[teste] = pipeline.predict_proba(matriz[teste])[:, 1]

    exigir(not np.isnan(y_score).any(), "sobrou predição vazia no Protocolo B")
    exigir((fold_b > 0).all(), "sobrou linha sem fold atribuído no Protocolo B")

    return pd.DataFrame({
        "protocolo": "B",
        "unidade": "imagem",
        "dataset": "bus-bra",
        "id": ids_tabela["id"],
        "paciente": ids_tabela["paciente"],
        "fold": fold_b,
        "y_true": y,
        "y_score": y_score,
    })[COLUNAS_SAIDA]


def verificar_sanidade(predicoes: pd.DataFrame) -> None:
    exigir(
        predicoes["y_score"].between(0, 1).all(),
        "há y_score fora do intervalo [0, 1] — não deveria acontecer com predict_proba",
    )
    exigir(not predicoes.isna().any().any(), "predicoes.csv ficou com valor vazio em alguma coluna")
    exigir(
        set(predicoes["y_true"].unique()) <= {0, 1},
        f"y_true com valor inesperado: {sorted(predicoes['y_true'].unique())}",
    )


def numero(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def resumir(predicoes: pd.DataFrame) -> None:
    for (protocolo, unidade), parte in predicoes.groupby(["protocolo", "unidade"], sort=True):
        print(f"  protocolo {protocolo}, unidade {unidade}: {numero(len(parte))} predições "
              f"({numero(int(parte['y_true'].sum()))} malignas)")


def main() -> None:
    matriz, ids_tabela = ler_caracteristicas()

    print(f"treinando com {matriz.shape[1]} características, {len(ids_tabela)} imagens...")

    print("Protocolo A (folds oficiais, agrupados por paciente)...")
    predicoes_a = rodar_protocolo_a(matriz, ids_tabela)

    print("Protocolo B (split aleatório por imagem, com vazamento proposital)...")
    predicoes_b = rodar_protocolo_b(matriz, ids_tabela)

    predicoes = pd.concat([predicoes_a, predicoes_b], ignore_index=True)
    verificar_sanidade(predicoes)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    predicoes.to_csv(SAIDA, index=False)

    print(f"\npredições salvas em {SAIDA.relative_to(RAIZ_CODIGO.parent).as_posix()}")
    resumir(predicoes)


if __name__ == "__main__":
    # O console do Windows costuma abrir em cp1252 e comeria os acentos das mensagens.
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
