"""Etapa 7 do pipeline: figura da curva ROC do Protocolo A (por paciente).

Le so `resultados/metricas.md` (bloco DADOS_MAQUINA, com os pontos de operacao) e
`resultados/roc.csv`, sem recalcular nada. Escreve `resultados/curva_roc.png`.
`protocolo_c/relatorio_breast.py` reaproveita `desenhar_roc` para o BrEaST.

Uso:
    python src/protocolo_ab/relatorio.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sem tela: precisa rodar igual num servidor/terminal sem display
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from comum.indexar import exigir

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RESULTADOS = RAIZ_CODIGO / "resultados"

CORES = {"0.5": "#d62728", "youden": "#2ca02c", "sens90": "#9467bd"}
MARCADORES = {"0.5": "o", "youden": "s", "sens90": "^"}


def desenhar_roc(entrada_md: Path, entrada_roc: Path, saida_png: Path, chave: str,
                 cor: str, rotulo_curva: str, titulo: str, nomes: dict[str, str]) -> None:
    """Desenha a curva ROC do combo `chave` ("protocolo|unidade") com os pontos de operacao."""
    correspondencia = re.search(r"<!-- DADOS_MAQUINA: (.+?) -->",
                                entrada_md.read_text(encoding="utf-8"), re.DOTALL)
    exigir(correspondencia is not None, f"não achei o bloco DADOS_MAQUINA em {entrada_md.name}")
    dados_maquina = json.loads(correspondencia.group(1))
    exigir(chave in dados_maquina, f"combo {chave} não está no bloco DADOS_MAQUINA")

    protocolo, unidade = chave.split("|")
    roc = pd.read_csv(entrada_roc)
    curva = roc[(roc["protocolo"] == protocolo) & (roc["unidade"] == unidade)].sort_values("fpr")

    fig, eixo = plt.subplots(figsize=(6, 6))
    eixo.plot(curva["fpr"], curva["tpr"], color=cor, linewidth=2, label=rotulo_curva)
    eixo.plot([0, 1], [0, 1], color="#999999", linewidth=1, linestyle="--",
              label="acaso (AUROC = 0,5)")

    for nome, ponto in dados_maquina[chave].items():
        eixo.scatter(
            ponto["fpr"], ponto["tpr"],
            color=CORES.get(nome, "black"), marker=MARCADORES.get(nome, "x"),
            s=90, zorder=5, edgecolor="white", linewidth=0.8,
            label=f"{nomes.get(nome, nome)} "
                  f"(sens={ponto['tpr']:.2f}, espec={1 - ponto['fpr']:.2f})",
        )

    eixo.set_xlabel("1 - especificidade (falso-positivo)")
    eixo.set_ylabel("sensibilidade (verdadeiro-positivo)")
    eixo.set_title(titulo)
    eixo.set_xlim(-0.02, 1.02)
    eixo.set_ylim(-0.02, 1.02)
    eixo.legend(loc="lower right", fontsize=8)
    eixo.grid(True, alpha=0.3)
    fig.tight_layout()

    saida_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida_png, dpi=150)
    plt.close(fig)
    print(f"figura salva em {saida_png.relative_to(RAIZ_CODIGO.parent).as_posix()}")


def main() -> None:
    desenhar_roc(
        RESULTADOS / "metricas.md", RESULTADOS / "roc.csv", RESULTADOS / "curva_roc.png",
        chave="A|paciente",
        cor="#1f5aa6",
        rotulo_curva="curva ROC (Protocolo A, por paciente)",
        titulo="Curva ROC — Protocolo A (folds oficiais, agregado por paciente)",
        nomes={"0.5": "limiar 0,5 (fixo)", "youden": "limiar de Youden",
               "sens90": "sensibilidade >= 0,90"},
    )


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
