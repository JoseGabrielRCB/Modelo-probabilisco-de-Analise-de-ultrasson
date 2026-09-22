"""Métricas do Protocolo C (BrEaST — validação externa), a partir de
`resultados/breast/predicoes_breast.csv`.

Reaproveita de `metricas.py` (import, não cópia — mesmo diretório src/) a maquinaria de
bootstrap por paciente e as funções de limiar: `ic95_bootstrap`,
`gerar_reamostragens_por_paciente`, `limiar_youden`, `maior_limiar_sensibilidade_minima`,
`fmt`, `fmt_ic`. Diferença central em relação a `metricas.py`: lá, os limiares (Youden,
sensibilidade>=0,90) são escolhidos fold-a-fold, fora de fold, dentro do próprio BUS-BRA;
aqui não há fold nenhum no BrEaST — os DOIS limiares são CONGELADOS, calculados uma única
vez sobre o BUS-BRA inteiro (Protocolo A, unidade=paciente, as 1.064 predições
fora-de-fold agregadas por paciente) e aplicados sem reajuste a TODAS as 252 predições do
BrEaST. É a mesma conta que `experimento_c.py` já documenta no log — refeita aqui a
partir do mesmo `predicoes.csv` (cálculo determinístico, os dois scripts sempre
concordam), porque quem de fato classifica os escores do BrEaST nos três pontos de
operação (0,5 / Youden / sensibilidade>=0,90) é este script, não `experimento_c.py`.

Calcula, para o combo (protocolo=C, unidade=paciente, dataset=breast):
    - AUROC e AUPRC, com IC95% por bootstrap (1.000 repetições, reamostrando PACIENTES,
      percentis 2,5/97,5 — mesmo método de metricas.py);
    - nos três pontos de operação (limiar 0,5 fixo, limiar de Youden congelado, limiar de
      sensibilidade>=0,90 congelado): sensibilidade, especificidade, acurácia e matriz de
      confusão, todos com IC95%.

Inclui uma seção curta comparando explicitamente com o BUS-BRA interno (AUROC do
Protocolo A / paciente, recalculado do mesmo `resultados/predicoes.csv` — nunca
hard-coded — para nunca poder divergir do que já está em `resultados/metricas.md`): é o
número mais importante do Protocolo C para o TCC, a "queda" (ou não) da validação
externa.

Escreve:
    - `resultados/breast/metricas_breast.md`
    - `resultados/breast/roc_breast.csv`

Uso:
    python src/protocolo_c/metricas_breast.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Reaproveita a maquinaria de bootstrap/limiar de metricas.py — pacote
# `src/protocolo_ab/`, ver docstring do módulo. `metricas.py` não é modificado por
# este import.
#
# Desde a reorganização de src/ por trilha (comum/, protocolo_ab/, protocolo_c/,
# resnet/), os módulos reaproveitados ficam em outro pacote: `src/` entra no sys.path
# para que o(s) import(s) abaixo funcionem rodando o script direto, de qualquer
# diretório. Só muda onde o Python procura o módulo — nenhuma lógica é alterada.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from comum.indexar import exigir
from protocolo_ab.metricas import (
    N_BOOTSTRAP,
    SEMENTE,
    SENSIBILIDADE_MINIMA,
    fmt,
    fmt_ic,
    gerar_reamostragens_por_paciente,
    ic95_bootstrap,
    limiar_youden,
    maior_limiar_sensibilidade_minima,
)

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo

ENTRADA_PREDICOES_BREAST = RAIZ_CODIGO / "resultados" / "breast" / "predicoes_breast.csv"
ENTRADA_PREDICOES_BUSBRA = RAIZ_CODIGO / "resultados" / "predicoes.csv"

SAIDA_MD = RAIZ_CODIGO / "resultados" / "breast" / "metricas_breast.md"
SAIDA_ROC = RAIZ_CODIGO / "resultados" / "breast" / "roc_breast.csv"

NOME_PONTO_SENS = f"sens{int(SENSIBILIDADE_MINIMA * 100)}"
NOMES_LEGIVEIS = {
    "0.5": "limiar 0,5 (fixo)",
    "youden": "limiar de Youden (congelado no BUS-BRA)",
    NOME_PONTO_SENS: f"sensibilidade >= {fmt(SENSIBILIDADE_MINIMA, 2)} (congelado no BUS-BRA)",
}


def ler_predicoes_breast() -> pd.DataFrame:
    exigir(
        ENTRADA_PREDICOES_BREAST.exists(),
        f"não encontrei {ENTRADA_PREDICOES_BREAST}; rode experimento_c.py antes",
    )
    tabela = pd.read_csv(ENTRADA_PREDICOES_BREAST)
    colunas_esperadas = {"protocolo", "unidade", "dataset", "id", "paciente", "y_true", "y_score"}
    exigir(
        colunas_esperadas.issubset(tabela.columns),
        f"faltam colunas em {ENTRADA_PREDICOES_BREAST.name}: {colunas_esperadas - set(tabela.columns)}",
    )
    return tabela


def ler_predicoes_busbra_a_paciente() -> pd.DataFrame:
    """As 1.064 predições fora-de-fold do Protocolo A, já agregadas por paciente — usadas
    tanto para congelar os limiares quanto para a comparação interno x externo."""
    exigir(
        ENTRADA_PREDICOES_BUSBRA.exists(),
        f"não encontrei {ENTRADA_PREDICOES_BUSBRA}; rode experimento.py do BUS-BRA antes",
    )
    predicoes = pd.read_csv(ENTRADA_PREDICOES_BUSBRA)
    parte = predicoes[(predicoes["protocolo"] == "A") & (predicoes["unidade"] == "paciente")]
    exigir(not parte.empty, "não achei protocolo=A, unidade=paciente em predicoes.csv do BUS-BRA")
    return parte


def calcular_limiares_congelados(predicoes_busbra_paciente: pd.DataFrame) -> tuple[float, float]:
    """Mesma conta de `experimento_c.calcular_limiares_congelados` — refeita aqui a
    partir do mesmo `predicoes.csv` (determinística, sempre concorda)."""
    y_true = predicoes_busbra_paciente["y_true"].to_numpy()
    y_score = predicoes_busbra_paciente["y_score"].to_numpy()
    limiar_y = limiar_youden(y_true, y_score)
    limiar_s90 = maior_limiar_sensibilidade_minima(y_true, y_score, SENSIBILIDADE_MINIMA)
    return limiar_y, limiar_s90


def metricas_do_breast(df: pd.DataFrame, limiar_y: float, limiar_s90: float) -> dict:
    """Calcula AUROC/AUPRC + pontos de operação para o BrEaST, usando os DOIS limiares já
    congelados no BUS-BRA (não reotimizados aqui)."""
    df = df.reset_index(drop=True)
    y_true = df["y_true"].to_numpy()
    y_score = df["y_score"].to_numpy()
    paciente = df["paciente"].to_numpy()

    pontos = {
        "0.5": (y_score >= 0.5).astype(int),
        "youden": (y_score >= limiar_y).astype(int),
        NOME_PONTO_SENS: (y_score >= limiar_s90).astype(int),
    }

    reamostragens = gerar_reamostragens_por_paciente(paciente, N_BOOTSTRAP, SEMENTE)

    resultado: dict = {"n_linhas": len(df), "n_pacientes": len(np.unique(paciente))}

    resultado["auroc"] = float(roc_auc_score(y_true, y_score))
    resultado["auroc_ic"] = ic95_bootstrap(roc_auc_score, reamostragens, y_true, y_score)
    resultado["auprc"] = float(average_precision_score(y_true, y_score))
    resultado["auprc_ic"] = ic95_bootstrap(average_precision_score, reamostragens, y_true, y_score)

    resultado["pontos_operacao"] = {}
    for nome, pred in pontos.items():
        sensibilidade = recall_score(y_true, pred, pos_label=1, zero_division=0)
        especificidade = recall_score(y_true, pred, pos_label=0, zero_division=0)
        acuracia = accuracy_score(y_true, pred)
        matriz = confusion_matrix(y_true, pred, labels=[0, 1])  # [[TN,FP],[FN,TP]]

        sens_ic = ic95_bootstrap(
            lambda yt, yp: recall_score(yt, yp, pos_label=1, zero_division=0),
            reamostragens, y_true, pred,
        )
        espec_ic = ic95_bootstrap(
            lambda yt, yp: recall_score(yt, yp, pos_label=0, zero_division=0),
            reamostragens, y_true, pred,
        )
        acc_ic = ic95_bootstrap(accuracy_score, reamostragens, y_true, pred)

        resultado["pontos_operacao"][nome] = {
            "sensibilidade": float(sensibilidade), "sensibilidade_ic": sens_ic,
            "especificidade": float(especificidade), "especificidade_ic": espec_ic,
            "acuracia": float(acuracia), "acuracia_ic": acc_ic,
            "tn": int(matriz[0, 0]), "fp": int(matriz[0, 1]),
            "fn": int(matriz[1, 0]), "tp": int(matriz[1, 1]),
            "fpr": float(1 - especificidade), "tpr": float(sensibilidade),
        }

    return resultado


def metricas_busbra_interno(predicoes_busbra_paciente: pd.DataFrame) -> dict:
    """AUROC/AUPRC do Protocolo A/paciente do BUS-BRA, recalculado aqui (nunca
    hard-coded) SÓ para a seção de comparação — não recalcula os pontos de operação
    completos de metricas.md, isso já está lá."""
    y_true = predicoes_busbra_paciente["y_true"].to_numpy()
    y_score = predicoes_busbra_paciente["y_score"].to_numpy()
    paciente = predicoes_busbra_paciente["paciente"].to_numpy()
    reamostragens = gerar_reamostragens_por_paciente(paciente, N_BOOTSTRAP, SEMENTE)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auroc_ic": ic95_bootstrap(roc_auc_score, reamostragens, y_true, y_score),
    }


def montar_markdown(r_breast: dict, r_busbra: dict, limiar_y: float, limiar_s90: float) -> str:
    linhas = ["# Métricas do BrEaST — Protocolo C (validação externa)", ""]
    linhas.append(
        "Modelo treinado UMA vez no BUS-BRA inteiro (sem validação cruzada) e aplicado "
        "congelado ao BrEaST — nenhum ajuste foi feito olhando o BrEaST. Os dois limiares "
        "de decisão abaixo também são congelados: calculados só sobre o BUS-BRA "
        "(Protocolo A, unidade=paciente, 1.064 predições fora-de-fold agregadas por "
        "paciente) e aplicados sem reajuste aqui — ver `experimento_c.py`/`metricas_breast.py`."
    )
    linhas.append("")
    linhas.append(f"- limiar de Youden (congelado no BUS-BRA): **{fmt(limiar_y, 4)}**")
    linhas.append(
        f"- maior limiar com sensibilidade >= {fmt(SENSIBILIDADE_MINIMA, 2)} "
        f"(congelado no BUS-BRA): **{fmt(limiar_s90, 4)}**"
    )
    linhas.append("")

    linhas.append(f"IC95% por bootstrap (n={N_BOOTSTRAP}), reamostrando pacientes.")
    linhas.append("")
    linhas.append(
        f"{r_breast['n_linhas']} pacientes do BrEaST "
        f"({r_breast['n_pacientes']} pacientes distintos — 1 imagem por paciente, sem "
        "o conceito de duas vistas do BUS-BRA)."
    )
    linhas.append("")

    linhas.append("| métrica | valor | IC95% |")
    linhas.append("|---|---|---|")
    linhas.append(f"| AUROC | {fmt(r_breast['auroc'])} | [{fmt(r_breast['auroc_ic'][0])}, {fmt(r_breast['auroc_ic'][1])}] |")
    linhas.append(f"| AUPRC | {fmt(r_breast['auprc'])} | [{fmt(r_breast['auprc_ic'][0])}, {fmt(r_breast['auprc_ic'][1])}] |")
    linhas.append("")

    linhas.append("Pontos de operação (limiar 0,5 fixo; Youden e sensibilidade>=0,90 CONGELADOS no BUS-BRA):")
    linhas.append("")
    linhas.append("| ponto | sensibilidade | especificidade | acurácia | TP | TN | FP | FN |")
    linhas.append("|---|---|---|---|---|---|---|---|")
    for nome, p in r_breast["pontos_operacao"].items():
        linhas.append(
            f"| {NOMES_LEGIVEIS.get(nome, nome)} "
            f"| {fmt_ic(p['sensibilidade'], p['sensibilidade_ic'])} "
            f"| {fmt_ic(p['especificidade'], p['especificidade_ic'])} "
            f"| {fmt_ic(p['acuracia'], p['acuracia_ic'])} "
            f"| {p['tp']} | {p['tn']} | {p['fp']} | {p['fn']} |"
        )
    linhas.append("")

    linhas.append("## Comparação: validação interna (BUS-BRA) x validação externa (BrEaST)")
    linhas.append("")
    linhas.append(
        "AUROC do BUS-BRA recalculado aqui a partir do mesmo `resultados/predicoes.csv` "
        "(Protocolo A, unidade=paciente) — nunca copiado à mão, para nunca poder divergir "
        "do que está em `resultados/metricas.md`."
    )
    linhas.append("")
    diferenca = r_breast["auroc"] - r_busbra["auroc"]
    linhas.append("| base | AUROC | IC95% |")
    linhas.append("|---|---|---|")
    linhas.append(
        f"| BUS-BRA (interno, CV por paciente, Protocolo A) "
        f"| {fmt(r_busbra['auroc'])} | [{fmt(r_busbra['auroc_ic'][0])}, {fmt(r_busbra['auroc_ic'][1])}] |"
    )
    linhas.append(
        f"| BrEaST (externo, congelado, Protocolo C) "
        f"| {fmt(r_breast['auroc'])} | [{fmt(r_breast['auroc_ic'][0])}, {fmt(r_breast['auroc_ic'][1])}] |"
    )
    linhas.append(f"| diferença (BrEaST − BUS-BRA) | {fmt(diferenca)} | — |")
    linhas.append("")
    linhas.append(
        f"**Este é o número mais importante do Protocolo C**: uma AUROC "
        f"{'mais baixa' if diferenca < 0 else 'mais alta ou igual'} no BrEaST em relação "
        "ao BUS-BRA é o resultado esperado ao trocar de hospital/aparelho sem re-treinar "
        "— o tamanho exato dessa queda (ou não-queda) é reportado aqui sem maquiagem, "
        "seja ela grande ou pequena."
    )
    linhas.append("")

    linhas.append("---")
    linhas.append("")
    linhas.append(
        "Bloco de dados de máquina (não editar à mão — mesmo padrão de `metricas.py`, "
        "para uma futura `relatorio_breast.py` poder marcar os pontos de operação na "
        "curva ROC sem reabrir `predicoes_breast.csv`):"
    )
    linhas.append("")
    dados_maquina = {
        "C|paciente": {
            nome: {"fpr": p["fpr"], "tpr": p["tpr"]}
            for nome, p in r_breast["pontos_operacao"].items()
        }
    }
    linhas.append(f"<!-- DADOS_MAQUINA: {json.dumps(dados_maquina)} -->")
    linhas.append("")

    return "\n".join(linhas)


def montar_roc_csv(df: pd.DataFrame) -> pd.DataFrame:
    fpr, tpr, limiares = roc_curve(df["y_true"], df["y_score"])
    return pd.DataFrame({
        "protocolo": "C", "unidade": "paciente", "dataset": "breast",
        "fpr": fpr, "tpr": tpr, "limiar": limiares,
    })


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    predicoes_breast = ler_predicoes_breast()
    predicoes_busbra_paciente = ler_predicoes_busbra_a_paciente()

    limiar_y, limiar_s90 = calcular_limiares_congelados(predicoes_busbra_paciente)
    print(f"limiar de Youden (congelado no BUS-BRA): {limiar_y:.4f}")
    print(f"limiar de sensibilidade>={SENSIBILIDADE_MINIMA:.2f} (congelado no BUS-BRA): {limiar_s90:.4f}")

    print(f"calculando métricas do BrEaST ({len(predicoes_breast)} pacientes)...")
    r_breast = metricas_do_breast(predicoes_breast, limiar_y, limiar_s90)

    print("calculando AUROC interno do BUS-BRA para comparação (Protocolo A, paciente)...")
    r_busbra = metricas_busbra_interno(predicoes_busbra_paciente)

    SAIDA_MD.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_MD.write_text(montar_markdown(r_breast, r_busbra, limiar_y, limiar_s90), encoding="utf-8")

    roc = montar_roc_csv(predicoes_breast)
    roc.to_csv(SAIDA_ROC, index=False)

    print(f"\nmétricas salvas em {SAIDA_MD.relative_to(RAIZ_CODIGO.parent).as_posix()}")
    print(f"curva ROC salva em {SAIDA_ROC.relative_to(RAIZ_CODIGO.parent).as_posix()}")

    diferenca = r_breast["auroc"] - r_busbra["auroc"]
    print(
        f"\n  AUROC BUS-BRA (interno) = {fmt(r_busbra['auroc'])} | "
        f"AUROC BrEaST (externo) = {fmt(r_breast['auroc'])} | "
        f"diferença = {fmt(diferenca)}"
    )


if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
