"""Etapa 6 do caminho ResNet18: as MESMAS metricas de `protocolo_ab/metricas.py` (bootstrap
por paciente, limiares fora-de-fold, alerta de sanidade), a partir de
`resultados/predicoes_resnet.csv`. Escreve `metricas_resnet.md` e `roc_resnet.csv`.

Uso:
    python src/resnet/metricas_resnet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from protocolo_ab.metricas import N_BOOTSTRAP, RAIZ_CODIGO, main

RESULTADOS = RAIZ_CODIGO / "resultados"

TEXTOS_MD_RESNET = {
    "cabecalho": [
        "# Métricas do experimento — caminho ResNet18 (comparação)",
        "",
        "Características extraídas por `features_resnet.py` (ResNet18 pré-treinada em "
        "ImageNet, penúltima camada, 512-d) em vez das características clássicas de "
        "`features.py` -- ver `resultados/metricas.md` para o resultado clássico "
        "(scikit-image + regressão logística). MESMO classificador, MESMOS folds, MESMA "
        "semente nos dois caminhos; só a extração de características muda.",
        "",
        f"IC95% por bootstrap (n={N_BOOTSTRAP}), reamostrando pacientes — ver "
        "`metricas_resnet.py` (cópia adaptada de `metricas.py`) para o método completo.",
        "",
    ],
    "pontos": "Pontos de operação (limiares Youden e sensibilidade>=0,90 escolhidos "
              "fora de fold — ver docstring de metricas.py):",
    "maquina": "Bloco de dados de máquina (não editar à mão; mesmo formato de `metricas.py`, "
               "reservado para uso futuro por um `relatorio_resnet.py`, que não existe ainda):",
}

if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main(RESULTADOS / "predicoes_resnet.csv", RESULTADOS / "metricas_resnet.md",
             RESULTADOS / "roc_resnet.csv", TEXTOS_MD_RESNET)
    except ValueError as erro:
        # Usar esse metodo é um pouco estranho , mas foi nescessario para poder validar um erro ocorrido e tambem por questao de docuemntao
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
