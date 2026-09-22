"""Etapa 5 do caminho ResNet18: treina o classificador e coleta predições fora-de-amostra
a partir das características extraídas por `features_resnet.py`.

Cópia adaptada de `experimento.py` (caminho clássico) -- MESMA lógica, MESMOS protocolos,
MESMA semente, único ponto que muda é a fonte das características (ResNet18 em vez de
scikit-image) e o nome do arquivo de saída (`predicoes_resnet.csv`, nunca
`predicoes.csv`, para não sobrescrever o resultado clássico já verificado). Ver
`experimento.py` para a documentação completa da lógica de classificação e dos dois
protocolos -- não repetida aqui em detalhe para não divergir das duas cópias com o tempo;
qualquer mudança de fundo na lógica de classificação deve ser espelhada manualmente nos
dois arquivos.

Lê `dados_processados/caracteristicas_resnet.npy` e
`dados_processados/caracteristicas_resnet_ids.csv` (saída de `features_resnet.py`).

Classificador: `Pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
class_weight="balanced"))` -- o mesmo do caminho clássico, aplicado agora sobre um vetor
de 512 características (ResNet18) em vez das ~62 características clássicas. Semente fixa
42 em tudo que aceita `random_state`.

Dois protocolos, o mesmo classificador nos dois:
    Protocolo A (principal) -- validação cruzada pelos 5 folds oficiais (coluna `fold`,
    agrupados por paciente). Depois, agrega por paciente (probabilidade média das
    imagens do mesmo paciente).
    Protocolo B (vazamento, só para comparação) -- StratifiedKFold aleatório por imagem,
    ignorando de propósito o agrupamento por paciente.

Escreve `resultados/predicoes_resnet.csv`, com uma linha por (protocolo, unidade,
observação): protocolo (A ou B), unidade (imagem ou paciente), dataset (bus-bra), id,
paciente, fold, y_true, y_score.

Uso:
    python src/resnet/experimento_resnet.py
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
# Caminhos -- tudo derivado da posição deste arquivo (ver indexar.py).
# ---------------------------------------------------------------------------

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo

ENTRADA_MATRIZ = RAIZ_CODIGO / "dados_processados" / "caracteristicas_resnet.npy"
ENTRADA_IDS = RAIZ_CODIGO / "dados_processados" / "caracteristicas_resnet_ids.csv"

# Nome diferente do caminho clássico -- NUNCA sobrescrever resultados/predicoes.csv.
SAIDA = RAIZ_CODIGO / "resultados" / "predicoes_resnet.csv"

SEMENTE = 42
N_FOLDS_PROTOCOLO_B = 5
COLUNAS_SAIDA = ["protocolo", "unidade", "dataset", "id", "paciente", "fold", "y_true", "y_score"]


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com uma mensagem clara se a condição não valer."""
    if not condicao:
        raise ValueError(mensagem)


def ler_caracteristicas() -> tuple[np.ndarray, pd.DataFrame]:
    exigir(
        ENTRADA_MATRIZ.exists(),
        f"não encontrei {ENTRADA_MATRIZ}; rode features_resnet.py antes",
    )
    exigir(
        ENTRADA_IDS.exists(),
        f"não encontrei {ENTRADA_IDS}; rode features_resnet.py antes",
    )

    matriz = np.load(ENTRADA_MATRIZ)
    ids_tabela = pd.read_csv(ENTRADA_IDS)

    exigir(
        matriz.shape[0] == len(ids_tabela),
        f"matriz tem {matriz.shape[0]} linhas, tabela de ids tem {len(ids_tabela)} — "
        "features_resnet.py precisa ser rodado de novo (arquivos dessincronizados)",
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
    o efeito do vazamento (comparação feita em metricas_resnet.py), não é um resultado a
    favor do modelo."""
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
    exigir(not predicoes.isna().any().any(), "predicoes_resnet.csv ficou com valor vazio em alguma coluna")
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

    print(f"treinando com {matriz.shape[1]} características (ResNet18), {len(ids_tabela)} imagens...")

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
