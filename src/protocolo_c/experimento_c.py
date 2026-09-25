"""Protocolo C: treina no BUS-BRA inteiro (sem validacao cruzada) e aplica, congelado, ao
BrEaST — a validacao externa do TCC.

Um unico modelo e ajustado sobre as 1.875 imagens do BUS-BRA (a coluna `fold` nao e usada)
e aplicado sem nenhum ajuste ao BrEaST, que ele nunca viu. Nenhuma escolha (pesos,
padronizacao, limiares) olha para o BrEaST. Mesmo classificador do Protocolo A
(`novo_pipeline`, importado de `protocolo_ab/experimento.py`). Os limiares congelados sao
calculados e aplicados em `metricas_breast.py`.

Escreve `resultados/breast/predicoes_breast.csv`, uma linha por paciente do BrEaST:
protocolo ("C"), unidade ("paciente"), dataset ("breast"), id, paciente, y_true, y_score.

Uso:
    python src/protocolo_c/experimento_c.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from comum.indexar import numero
from protocolo_ab.experimento import novo_pipeline

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
DADOS = RAIZ_CODIGO / "dados_processados"

SAIDA = RAIZ_CODIGO / "resultados" / "breast" / "predicoes_breast.csv"

COLUNAS_SAIDA = ["protocolo", "unidade", "dataset", "id", "paciente", "y_true", "y_score"]


def main() -> None:
    matriz_busbra = np.load(DADOS / "caracteristicas.npy")
    ids_busbra = pd.read_csv(DADOS / "caracteristicas_ids.csv")
    matriz_breast = np.load(DADOS / "breast" / "caracteristicas_breast.npy")
    ids_breast = pd.read_csv(DADOS / "breast" / "caracteristicas_breast_ids.csv")

    print(
        f"treinando no BUS-BRA completo: "
        f"{numero(len(ids_busbra))} imagens, {numero(ids_busbra['paciente'].nunique())} pacientes"
    )
    pipeline = novo_pipeline()
    pipeline.fit(matriz_busbra, ids_busbra["rotulo"].to_numpy())

    predicoes = pd.DataFrame(
        {
            "protocolo": "C",
            "unidade": "paciente",
            "dataset": "breast",
            "id": ids_breast["id"],
            "paciente": ids_breast["paciente"],
            "y_true": ids_breast["rotulo"],
            "y_score": pipeline.predict_proba(matriz_breast)[:, 1],
        }
    )[COLUNAS_SAIDA]

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    predicoes.to_csv(SAIDA, index=False)

    print(f"\npredicoes do Protocolo C salvas em {SAIDA.relative_to(RAIZ_CODIGO.parent).as_posix()}")
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
        # Usar esse metodo é um pouco estranho , mas foi nescessario para poder validar um erro ocorrido e tambem por questao de docuemntao
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
