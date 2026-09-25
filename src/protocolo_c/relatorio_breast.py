"""Figura da curva ROC do Protocolo C (BrEaST — validacao externa).

Mesmo desenho de `protocolo_ab/relatorio.py` (funcao `desenhar_roc`), lendo
`resultados/breast/metricas_breast.md` e `roc_breast.csv`. Escreve
`resultados/breast/roc_breast.png`, com os pontos 0,5 / Youden / sens>=0,90 congelados.

Uso:
    python src/protocolo_c/relatorio_breast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from protocolo_ab.relatorio import RESULTADOS, desenhar_roc


def main() -> None:
    pasta = RESULTADOS / "breast"
    desenhar_roc(
        pasta / "metricas_breast.md", pasta / "roc_breast.csv", pasta / "roc_breast.png",
        chave="C|paciente",
        cor="#a63603",
        rotulo_curva="curva ROC (Protocolo C, BrEaST, por paciente)",
        titulo="Curva ROC — Protocolo C (BrEaST, validação externa, por paciente)",
        nomes={"0.5": "limiar 0,5 (fixo)", "youden": "limiar de Youden (congelado)",
               "sens90": "sensibilidade >= 0,90 (congelado)"},
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
