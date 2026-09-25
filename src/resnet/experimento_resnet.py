"""Etapa 5 do caminho ResNet18: o MESMO experimento de `protocolo_ab/experimento.py`
(mesmo classificador, mesmos protocolos A/B, mesma semente), aplicado as caracteristicas de
`features_resnet.py`. Escreve `resultados/predicoes_resnet.csv`, sem tocar em `predicoes.csv`.

Uso:
    python src/resnet/experimento_resnet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from protocolo_ab.experimento import RAIZ_CODIGO, main

DADOS = RAIZ_CODIGO / "dados_processados"

if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main(DADOS / "caracteristicas_resnet.npy", DADOS / "caracteristicas_resnet_ids.csv",
             RAIZ_CODIGO / "resultados" / "predicoes_resnet.csv")
    except ValueError as erro:
        # Usar esse metodo é um pouco estranho , mas foi nescessario para poder validar um erro ocorrido e tambem por questao de docuemntao
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
