"""Etapa 7 do pipeline: gera a figura da curva ROC do Protocolo A (por paciente).

Lê só `resultados/metricas.md` e `resultados/roc.csv` (saídas de `metricas.py`) — nunca
recalcula nada de `predicoes.csv` diretamente. Os pontos de operação (0,5, Youden,
sensibilidade >= 0,90) são lidos do bloco "DADOS_MAQUINA" embutido em `metricas.md` como
comentário HTML (ver o docstring de `metricas.py`, seção "Bloco de dados de máquina") —
evita ter que reimplementar aqui a lógica de limiar fora-de-fold, que já vive só em
`metricas.py`.

Escreve `resultados/curva_roc.png`: a curva ROC do Protocolo A / paciente, com os três
pontos de operação marcados e uma legenda.

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

# ---------------------------------------------------------------------------
# Caminhos — tudo derivado da posição deste arquivo (ver indexar.py).
# ---------------------------------------------------------------------------

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo

ENTRADA_MD = RAIZ_CODIGO / "resultados" / "metricas.md"
ENTRADA_ROC = RAIZ_CODIGO / "resultados" / "roc.csv"

SAIDA_PNG = RAIZ_CODIGO / "resultados" / "curva_roc.png"

PROTOCOLO_ALVO = "A"
UNIDADE_ALVO = "paciente"

NOMES_LEGIVEIS = {
    "0.5": "limiar 0,5 (fixo)",
    "youden": "limiar de Youden",
    "sens90": "sensibilidade >= 0,90",
}


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com uma mensagem clara se a condição não valer."""
    if not condicao:
        raise ValueError(mensagem)


def ler_dados_maquina() -> dict:
    """Extrai o bloco JSON `<!-- DADOS_MAQUINA: {...} -->` de metricas.md."""
    exigir(ENTRADA_MD.exists(), f"não encontrei {ENTRADA_MD}; rode metricas.py antes")
    conteudo = ENTRADA_MD.read_text(encoding="utf-8")

    correspondencia = re.search(r"<!-- DADOS_MAQUINA: (.+?) -->", conteudo, re.DOTALL)
    exigir(
        correspondencia is not None,
        f"não achei o bloco DADOS_MAQUINA em {ENTRADA_MD.name} — rode metricas.py de novo",
    )
    return json.loads(correspondencia.group(1))


def ler_curva_roc() -> pd.DataFrame:
    exigir(ENTRADA_ROC.exists(), f"não encontrei {ENTRADA_ROC}; rode metricas.py antes")
    roc = pd.read_csv(ENTRADA_ROC)
    parte = roc[(roc["protocolo"] == PROTOCOLO_ALVO) & (roc["unidade"] == UNIDADE_ALVO)]
    exigir(
        not parte.empty,
        f"não achei curva para protocolo={PROTOCOLO_ALVO}, unidade={UNIDADE_ALVO} em {ENTRADA_ROC.name}",
    )
    return parte.sort_values("fpr")


def montar_figura(curva: pd.DataFrame, pontos: dict) -> None:
    fig, eixo = plt.subplots(figsize=(6, 6))

    eixo.plot(curva["fpr"], curva["tpr"], color="#1f5aa6", linewidth=2,
              label="curva ROC (Protocolo A, por paciente)")
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
    eixo.set_title("Curva ROC — Protocolo A (folds oficiais, agregado por paciente)")
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
    chave = f"{PROTOCOLO_ALVO}|{UNIDADE_ALVO}"
    exigir(
        chave in dados_maquina,
        f"combo {chave} não está no bloco DADOS_MAQUINA de {ENTRADA_MD.name}",
    )
    pontos = dados_maquina[chave]

    curva = ler_curva_roc()
    montar_figura(curva, pontos)

    print(f"figura salva em {SAIDA_PNG.relative_to(RAIZ_CODIGO.parent).as_posix()}")


if __name__ == "__main__":
    # O console do Windows costuma abrir em cp1252 e comeria os acentos das mensagens.
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
