"""Figura da curva ROC do Protocolo C (BrEaST — validação externa).

Não fazia parte da lista original de scripts do Protocolo C, mas a pasta
`resultados/breast/` foi pedida explicitamente para conter "predições, métricas, FIGURA
do BrEaST" — este script fecha essa lista, no mesmo padrão de `relatorio.py` (Etapa 7 do
BUS-BRA): lê só `resultados/breast/metricas_breast.md` e
`resultados/breast/roc_breast.csv` (nunca recalcula nada de `predicoes_breast.csv`
diretamente), e escreve `resultados/breast/roc_breast.png` com a curva ROC do Protocolo C
e os três pontos de operação marcados (0,5 / Youden congelado / sensibilidade>=0,90
congelada).

Uso:
    python src/protocolo_c/relatorio_breast.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo

ENTRADA_MD = RAIZ_CODIGO / "resultados" / "breast" / "metricas_breast.md"
ENTRADA_ROC = RAIZ_CODIGO / "resultados" / "breast" / "roc_breast.csv"

SAIDA_PNG = RAIZ_CODIGO / "resultados" / "breast" / "roc_breast.png"

CHAVE_COMBO = "C|paciente"

NOMES_LEGIVEIS = {
    "0.5": "limiar 0,5 (fixo)",
    "youden": "limiar de Youden (congelado)",
    "sens90": "sensibilidade >= 0,90 (congelado)",
}


def exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise ValueError(mensagem)


def ler_dados_maquina() -> dict:
    exigir(ENTRADA_MD.exists(), f"não encontrei {ENTRADA_MD}; rode metricas_breast.py antes")
    conteudo = ENTRADA_MD.read_text(encoding="utf-8")
    correspondencia = re.search(r"<!-- DADOS_MAQUINA: (.+?) -->", conteudo, re.DOTALL)
    exigir(
        correspondencia is not None,
        f"não achei o bloco DADOS_MAQUINA em {ENTRADA_MD.name} — rode metricas_breast.py de novo",
    )
    return json.loads(correspondencia.group(1))


def ler_curva_roc() -> pd.DataFrame:
    exigir(ENTRADA_ROC.exists(), f"não encontrei {ENTRADA_ROC}; rode metricas_breast.py antes")
    roc = pd.read_csv(ENTRADA_ROC)
    return roc.sort_values("fpr")


def montar_figura(curva: pd.DataFrame, pontos: dict) -> None:
    fig, eixo = plt.subplots(figsize=(6, 6))

    eixo.plot(curva["fpr"], curva["tpr"], color="#a63603", linewidth=2,
              label="curva ROC (Protocolo C, BrEaST, por paciente)")
    eixo.plot([0, 1], [0, 1], color="#999999", linewidth=1, linestyle="--",
              label="acaso (AUROC = 0,5)")

    cores = {"0.5": "#d62728", "youden": "#2ca02c", "sens90": "#9467bd"}
    marcadores = {"0.5": "o", "youden": "s", "sens90": "^"}
    for nome, ponto in pontos.items():
        eixo.scatter(
            ponto["fpr"], ponto["tpr"],
            color=cores.get(nome, "black"), marker=marcadores.get(nome, "x"),
            s=90, zorder=5, edgecolor="white", linewidth=0.8,
            label=f"{NOMES_LEGIVEIS.get(nome, nome)} "
                  f"(sens={ponto['tpr']:.2f}, espec={1 - ponto['fpr']:.2f})",
        )

    eixo.set_xlabel("1 - especificidade (falso-positivo)")
    eixo.set_ylabel("sensibilidade (verdadeiro-positivo)")
    eixo.set_title("Curva ROC — Protocolo C (BrEaST, validação externa, por paciente)")
    eixo.set_xlim(-0.02, 1.02)
    eixo.set_ylim(-0.02, 1.02)
    eixo.legend(loc="lower right", fontsize=8)
    eixo.grid(True, alpha=0.3)
    fig.tight_layout()

    SAIDA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    dados_maquina = ler_dados_maquina()
    exigir(CHAVE_COMBO in dados_maquina, f"combo {CHAVE_COMBO} não está no bloco DADOS_MAQUINA")
    pontos = dados_maquina[CHAVE_COMBO]

    curva = ler_curva_roc()
    montar_figura(curva, pontos)

    print(f"figura salva em {SAIDA_PNG.relative_to(RAIZ_CODIGO.parent).as_posix()}")


if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
