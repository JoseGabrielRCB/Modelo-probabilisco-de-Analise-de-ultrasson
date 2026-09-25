"""Etapa 5 do pipeline: treina o classificador e grava as predicoes fora-de-amostra.

Le `caracteristicas.npy` / `caracteristicas_ids.csv` (saida de `features.py`) e escreve
`resultados/predicoes.csv` (protocolo, unidade, dataset, id, paciente, fold, y_true, y_score).

Classificador: StandardScaler + LogisticRegression(max_iter=2000, class_weight="balanced"),
semente 42. Um pipeline novo por fold, para o scaler nunca ver dados de teste.

    Protocolo A (principal): 5 folds oficiais, agrupados por paciente. Predicoes por imagem
    e tambem agregadas por paciente (media das probabilidades das imagens).
    Protocolo B (vazamento proposital): StratifiedKFold por imagem, ignorando o paciente.
    Existe so para medir, contra o A, o quanto o vazamento infla o desempenho.

`resnet/experimento_resnet.py` reaproveita este script com outros arquivos de entrada/saida.

Uso:
    python src/protocolo_ab/experimento.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from comum.indexar import exigir, numero

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo

ENTRADA_MATRIZ = RAIZ_CODIGO / "dados_processados" / "caracteristicas.npy"
ENTRADA_IDS = RAIZ_CODIGO / "dados_processados" / "caracteristicas_ids.csv"
SAIDA = RAIZ_CODIGO / "resultados" / "predicoes.csv"

SEMENTE = 42
N_FOLDS_PROTOCOLO_B = 5
COLUNAS_SAIDA = ["protocolo", "unidade", "dataset", "id", "paciente", "fold", "y_true", "y_score"]


def novo_pipeline() -> Pipeline:
    """Pipeline novo a cada chamada: reaproveitar um treinado vazaria o scaler entre folds."""
    return Pipeline([
        ("escala", StandardScaler()),
        ("logistica", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEMENTE,
        )),
    ])


def rodar_protocolo_a(matriz: np.ndarray, ids_tabela: pd.DataFrame) -> pd.DataFrame:
    """Validacao cruzada pelos 5 folds oficiais; devolve linhas por imagem e por paciente."""
    y = ids_tabela["rotulo"].to_numpy()
    fold = ids_tabela["fold"].to_numpy()

    y_score = np.full(len(y), np.nan)
    for f in sorted(np.unique(fold)):
        treino = fold != f
        teste = fold == f
        pipeline = novo_pipeline()
        pipeline.fit(matriz[treino], y[treino])
        y_score[teste] = pipeline.predict_proba(matriz[teste])[:, 1]

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

    # Agrega por paciente; fold e rotulo sao uniformes no paciente (ver particionar/verificar).
    agregado = linhas_imagem.groupby("paciente", as_index=False).agg(
        y_true=("y_true", "mean"),
        y_score=("y_score", "mean"),
        fold=("fold", "first"),
    )
    linhas_paciente = pd.DataFrame({
        "protocolo": "A",
        "unidade": "paciente",
        "dataset": "bus-bra",
        "id": agregado["paciente"].astype(str),  # na unidade paciente, id = paciente
        "paciente": agregado["paciente"],
        "fold": agregado["fold"],
        "y_true": agregado["y_true"].round().astype(int),
        "y_score": agregado["y_score"],
    })

    return pd.concat([linhas_imagem, linhas_paciente], ignore_index=True)[COLUNAS_SAIDA]


def rodar_protocolo_b(matriz: np.ndarray, ids_tabela: pd.DataFrame) -> pd.DataFrame:
    """Mesmos dados em StratifiedKFold aleatorio por imagem (vazamento proposital)."""
    y = ids_tabela["rotulo"].to_numpy()

    fold_b = np.zeros(len(y), dtype=int)
    y_score = np.full(len(y), np.nan)

    divisor = StratifiedKFold(n_splits=N_FOLDS_PROTOCOLO_B, shuffle=True, random_state=SEMENTE)
    for numero_do_fold, (treino, teste) in enumerate(divisor.split(matriz, y), start=1):
        fold_b[teste] = numero_do_fold
        pipeline = novo_pipeline()
        pipeline.fit(matriz[treino], y[treino])
        y_score[teste] = pipeline.predict_proba(matriz[teste])[:, 1]

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


def resumir(predicoes: pd.DataFrame) -> None:
    for (protocolo, unidade), parte in predicoes.groupby(["protocolo", "unidade"], sort=True):
        print(f"  protocolo {protocolo}, unidade {unidade}: {numero(len(parte))} predicoes "
              f"({numero(int(parte['y_true'].sum()))} malignas)")


def main(entrada_matriz: Path = ENTRADA_MATRIZ, entrada_ids: Path = ENTRADA_IDS,
         saida: Path = SAIDA) -> None:
    matriz = np.load(entrada_matriz)
    ids_tabela = pd.read_csv(entrada_ids)
    exigir(
        matriz.shape[0] == len(ids_tabela),
        f"matriz tem {matriz.shape[0]} linhas, tabela de ids tem {len(ids_tabela)} — "
        "rode a extração de características de novo (arquivos dessincronizados)",
    )

    print(f"treinando: {len(ids_tabela)} imagens, {matriz.shape[1]} carct")
    predicoes_a = rodar_protocolo_a(matriz, ids_tabela)
    predicoes_b = rodar_protocolo_b(matriz, ids_tabela)

    predicoes = pd.concat([predicoes_a, predicoes_b], ignore_index=True)
    saida.parent.mkdir(parents=True, exist_ok=True)
    predicoes.to_csv(saida, index=False)

    print(f"\npredicoes salvas em {saida.relative_to(RAIZ_CODIGO.parent).as_posix()}")
    resumir(predicoes)


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
