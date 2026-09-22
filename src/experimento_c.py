"""Protocolo C: treina no BUS-BRA inteiro (sem validação cruzada) e avalia, congelado, no
BrEaST — a validação externa do TCC.

Diferença central em relação ao Protocolo A (`experimento.py`): ali, o modelo é retreinado
5 vezes (uma por fold) e cada linha do BUS-BRA recebe uma predição fora-de-amostra DENTRO
do próprio BUS-BRA. Aqui, o BUS-BRA inteiro (as 1.875 imagens, sem olhar a coluna `fold`)
vira treino de UM ÚNICO modelo, e esse modelo — já congelado, sem nenhum ajuste adicional
— é aplicado ao BrEaST, que ele nunca viu. O objetivo é medir se o modelo generaliza para
aparelhos/hospital diferentes dos que apareceram no treino, não otimizar desempenho no
BrEaST: nenhuma escolha feita aqui (hiperparâmetro, limiar, ou qualquer outra coisa) olha
para os dados do BrEaST antes da predição final.

Mesmo classificador do Protocolo A, reaproveitado via import de `experimento.py`
(`novo_pipeline`) para os dois nunca divergirem: `Pipeline(StandardScaler(),
LogisticRegression(max_iter=2000, class_weight="balanced"))`, semente 42.

Limiares de decisão (Youden e sensibilidade >= 0,90): CONGELADOS no BUS-BRA, nunca
recalculados olhando o BrEaST. Calculados aqui a partir de TODAS as predições fora-de-fold
do Protocolo A já agregadas por paciente (`resultados/predicoes.csv`, protocolo=A,
unidade=paciente — as 1.064 linhas), usando as mesmas funções `limiar_youden` e
`maior_limiar_sensibilidade_minima` de `metricas.py` (reaproveitadas via import, não
copiadas, para não haver dois lugares onde a mesma conta poderia divergir). Diferença
proposital em relação a `metricas.py`: lá, o limiar é escolhido fold-a-fold ("fora de
fold"); aqui não há necessidade disso — é UM único limiar fixo, escolhido de uma vez sobre
todo o BUS-BRA, exatamente porque vai ser aplicado sem reajuste a uma base nova. Os dois
valores calculados são só impressos no log deste script (documentação); quem de fato
aplica esses limiares às predições do BrEaST é `metricas_breast.py`, que recalcula os
mesmos dois números a partir do mesmo `predicoes.csv` (cálculo determinístico, então os
dois scripts sempre concordam).

Escreve `resultados/breast/predicoes_breast.csv`, uma linha por paciente do BrEaST:
    protocolo (sempre "C"), unidade (sempre "paciente" — o BrEaST não tem o conceito de
    duas vistas por paciente do BUS-BRA), dataset (sempre "breast"), id, paciente,
    y_true, y_score.

Uso:
    python src/experimento_c.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reaproveita a arquitetura do classificador do Protocolo A e as funções de limiar de
# metricas.py — mesmo diretório src/, ver docstring do módulo. Nenhum dos dois módulos é
# modificado por este import (os dois só rodam `main()` sob `if __name__ == "__main__"`).
from experimento import novo_pipeline
from indexar import exigir
from metricas import SENSIBILIDADE_MINIMA, limiar_youden, maior_limiar_sensibilidade_minima

RAIZ_CODIGO = Path(__file__).resolve().parents[1]   # .../TCC-Ultrassom/03-codigo

ENTRADA_MATRIZ_BUSBRA = RAIZ_CODIGO / "dados_processados" / "caracteristicas.npy"
ENTRADA_IDS_BUSBRA = RAIZ_CODIGO / "dados_processados" / "caracteristicas_ids.csv"
ENTRADA_MATRIZ_BREAST = RAIZ_CODIGO / "dados_processados" / "breast" / "caracteristicas_breast.npy"
ENTRADA_IDS_BREAST = RAIZ_CODIGO / "dados_processados" / "breast" / "caracteristicas_breast_ids.csv"
ENTRADA_PREDICOES_BUSBRA = RAIZ_CODIGO / "resultados" / "predicoes.csv"

SAIDA = RAIZ_CODIGO / "resultados" / "breast" / "predicoes_breast.csv"

COLUNAS_SAIDA = ["protocolo", "unidade", "dataset", "id", "paciente", "y_true", "y_score"]

N_IMAGENS_BUSBRA_ESPERADO = 1875
N_PACIENTES_BUSBRA_PACIENTE_ESPERADO = 1064
N_PACIENTES_BREAST_ESPERADO = 252


def ler_caracteristicas_busbra() -> tuple[np.ndarray, pd.DataFrame]:
    """Lê o BUS-BRA INTEIRO (as 1.875 imagens) — a coluna `fold` não é usada aqui, não há
    validação cruzada no Protocolo C."""
    exigir(ENTRADA_MATRIZ_BUSBRA.exists(), f"não encontrei {ENTRADA_MATRIZ_BUSBRA}; rode features.py antes")
    exigir(ENTRADA_IDS_BUSBRA.exists(), f"não encontrei {ENTRADA_IDS_BUSBRA}; rode features.py antes")

    matriz = np.load(ENTRADA_MATRIZ_BUSBRA)
    ids_tabela = pd.read_csv(ENTRADA_IDS_BUSBRA)

    exigir(
        matriz.shape[0] == len(ids_tabela),
        f"matriz do BUS-BRA tem {matriz.shape[0]} linhas, tabela de ids tem {len(ids_tabela)}",
    )
    exigir(
        len(ids_tabela) == N_IMAGENS_BUSBRA_ESPERADO,
        f"esperava {N_IMAGENS_BUSBRA_ESPERADO} imagens do BUS-BRA, encontrei {len(ids_tabela)}",
    )
    return matriz, ids_tabela


def ler_caracteristicas_breast() -> tuple[np.ndarray, pd.DataFrame]:
    exigir(
        ENTRADA_MATRIZ_BREAST.exists(),
        f"não encontrei {ENTRADA_MATRIZ_BREAST}; rode indexar_breast.py, verificar_breast.py "
        "e features_breast.py antes",
    )
    exigir(ENTRADA_IDS_BREAST.exists(), f"não encontrei {ENTRADA_IDS_BREAST}")

    matriz = np.load(ENTRADA_MATRIZ_BREAST)
    ids_tabela = pd.read_csv(ENTRADA_IDS_BREAST)

    exigir(
        matriz.shape[0] == len(ids_tabela),
        f"matriz do BrEaST tem {matriz.shape[0]} linhas, tabela de ids tem {len(ids_tabela)}",
    )
    exigir(
        len(ids_tabela) == N_PACIENTES_BREAST_ESPERADO,
        f"esperava {N_PACIENTES_BREAST_ESPERADO} pacientes do BrEaST, encontrei {len(ids_tabela)}",
    )
    return matriz, ids_tabela


def calcular_limiares_congelados() -> tuple[float, float]:
    """Limiares de Youden e sensibilidade>=0,90, calculados SÓ sobre o BUS-BRA (Protocolo
    A, unidade=paciente, todas as 1.064 predições fora-de-fold agregadas por paciente) —
    ver docstring do módulo."""
    exigir(
        ENTRADA_PREDICOES_BUSBRA.exists(),
        f"não encontrei {ENTRADA_PREDICOES_BUSBRA}; rode experimento.py do BUS-BRA antes",
    )
    predicoes_busbra = pd.read_csv(ENTRADA_PREDICOES_BUSBRA)
    parte = predicoes_busbra[
        (predicoes_busbra["protocolo"] == "A") & (predicoes_busbra["unidade"] == "paciente")
    ]
    exigir(
        len(parte) == N_PACIENTES_BUSBRA_PACIENTE_ESPERADO,
        f"esperava {N_PACIENTES_BUSBRA_PACIENTE_ESPERADO} linhas em predicoes.csv "
        f"(protocolo=A, unidade=paciente), encontrei {len(parte)}",
    )

    y_true = parte["y_true"].to_numpy()
    y_score = parte["y_score"].to_numpy()

    limiar_y = limiar_youden(y_true, y_score)
    limiar_s90 = maior_limiar_sensibilidade_minima(y_true, y_score, SENSIBILIDADE_MINIMA)
    return limiar_y, limiar_s90


def verificar_sanidade(predicoes: pd.DataFrame) -> None:
    exigir(
        predicoes["y_score"].between(0, 1).all(),
        "há y_score fora do intervalo [0, 1] — não deveria acontecer com predict_proba",
    )
    exigir(not predicoes.isna().any().any(), "predicoes_breast.csv ficou com valor vazio em alguma coluna")
    exigir(
        set(predicoes["y_true"].unique()) <= {0, 1},
        f"y_true com valor inesperado: {sorted(predicoes['y_true'].unique())}",
    )
    exigir(
        len(predicoes) == N_PACIENTES_BREAST_ESPERADO,
        f"esperava {N_PACIENTES_BREAST_ESPERADO} linhas em predicoes_breast.csv, encontrei {len(predicoes)}",
    )


def numero(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def main() -> None:
    print("carregando características do BUS-BRA (treino, conjunto INTEIRO, sem CV)...")
    matriz_busbra, ids_busbra = ler_caracteristicas_busbra()

    print("carregando características do BrEaST (validação externa, nunca visto no treino)...")
    matriz_breast, ids_breast = ler_caracteristicas_breast()

    print(
        f"treinando UM classificador no BUS-BRA inteiro "
        f"({numero(len(ids_busbra))} imagens, {numero(ids_busbra['paciente'].nunique())} pacientes)..."
    )
    pipeline = novo_pipeline()
    pipeline.fit(matriz_busbra, ids_busbra["rotulo"].to_numpy())

    print(
        f"aplicando o modelo congelado ao BrEaST ({numero(len(ids_breast))} pacientes) — "
        "nenhum ajuste feito olhando o BrEaST..."
    )
    y_score_breast = pipeline.predict_proba(matriz_breast)[:, 1]

    print(
        "calculando limiares congelados no BUS-BRA (Protocolo A, unidade=paciente, "
        f"{N_PACIENTES_BUSBRA_PACIENTE_ESPERADO} predições fora-de-fold agregadas por paciente)..."
    )
    limiar_y, limiar_s90 = calcular_limiares_congelados()
    print(f"  limiar de Youden (congelado no BUS-BRA): {limiar_y:.4f}")
    print(f"  maior limiar com sensibilidade >= {SENSIBILIDADE_MINIMA:.2f} (congelado no BUS-BRA): {limiar_s90:.4f}")
    print(
        "  (estes dois valores são só documentados aqui; quem os aplica às predições do "
        "BrEaST é metricas_breast.py, recalculando-os do mesmo predicoes.csv)"
    )

    predicoes = pd.DataFrame(
        {
            "protocolo": "C",
            "unidade": "paciente",
            "dataset": "breast",
            "id": ids_breast["id"],
            "paciente": ids_breast["paciente"],
            "y_true": ids_breast["rotulo"],
            "y_score": y_score_breast,
        }
    )[COLUNAS_SAIDA]

    verificar_sanidade(predicoes)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    predicoes.to_csv(SAIDA, index=False)

    print(f"\npredições do Protocolo C salvas em {SAIDA.relative_to(RAIZ_CODIGO.parent).as_posix()}")
    print(
        f"  {numero(len(predicoes))} pacientes do BrEaST "
        f"({numero(int(predicoes['y_true'].sum()))} malignos)"
    )


if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
