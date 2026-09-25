"""Metricas do Protocolo C (BrEaST — validacao externa), a partir de
`resultados/breast/predicoes_breast.csv`.

Mesmo metodo de `protocolo_ab/metricas.py` (reaproveitado via import): AUROC/AUPRC e os 3
pontos de operacao, com IC95% por bootstrap de pacientes. Diferenca: nao ha fold no BrEaST,
entao os limiares de Youden e sens>=0,90 sao CONGELADOS — calculados uma unica vez sobre as
1.064 predicoes fora-de-fold do Protocolo A/paciente do BUS-BRA (`resultados/predicoes.csv`)
e aplicados sem reajuste as 252 predicoes do BrEaST.

Inclui a comparacao com a AUROC interna do BUS-BRA (Protocolo A/paciente), recalculada do
mesmo `predicoes.csv`: a "queda" da validacao externa.

Escreve `resultados/breast/metricas_breast.md` e `resultados/breast/roc_breast.csv`.

Uso:
    python src/protocolo_c/metricas_breast.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
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
    metricas_nos_pontos,
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


def ler_predicoes_busbra_a_paciente() -> pd.DataFrame:
    """As 1.064 predicoes fora do fold do Protocolo A, agregadas por paciente."""
    predicoes = pd.read_csv(ENTRADA_PREDICOES_BUSBRA)
    return predicoes[(predicoes["protocolo"] == "A") & (predicoes["unidade"] == "paciente")]


def metricas_do_breast(df: pd.DataFrame, limiar_y: float, limiar_s90: float) -> dict:
    """Metricas do BrEaST com os DOIS limiares congelados no BUS-BRA (sem reotimizar)."""
    df = df.reset_index(drop=True)
    y_true = df["y_true"].to_numpy()
    y_score = df["y_score"].to_numpy()
    pontos = {
        "0.5": (y_score >= 0.5).astype(int),
        "youden": (y_score >= limiar_y).astype(int),
        NOME_PONTO_SENS: (y_score >= limiar_s90).astype(int),
    }
    return metricas_nos_pontos(y_true, y_score, df["paciente"].to_numpy(), pontos)


def metricas_busbra_interno(predicoes_busbra_paciente: pd.DataFrame) -> dict:
    """AUROC (com IC95%) do Protocolo A/paciente do BUS-BRA, so para a comparacao."""
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

    predicoes_breast = pd.read_csv(ENTRADA_PREDICOES_BREAST)
    predicoes_busbra_paciente = ler_predicoes_busbra_a_paciente()

    # Limiares congelados: calculados so sobre o BUS-BRA (Protocolo A/paciente).
    y_true_busbra = predicoes_busbra_paciente["y_true"].to_numpy()
    y_score_busbra = predicoes_busbra_paciente["y_score"].to_numpy()
    limiar_y = limiar_youden(y_true_busbra, y_score_busbra)
    limiar_s90 = maior_limiar_sensibilidade_minima(y_true_busbra, y_score_busbra, SENSIBILIDADE_MINIMA)
    print(f"limiar de Youden (congelado no BUS-BRA): {limiar_y:.4f}")
    print(f"limiar de sensibilidade>={SENSIBILIDADE_MINIMA:.2f} (congelado no BUS-BRA): {limiar_s90:.4f}")

    r_breast = metricas_do_breast(predicoes_breast, limiar_y, limiar_s90)

    r_busbra = metricas_busbra_interno(predicoes_busbra_paciente)

    SAIDA_MD.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_MD.write_text(montar_markdown(r_breast, r_busbra, limiar_y, limiar_s90), encoding="utf-8")
    montar_roc_csv(predicoes_breast).to_csv(SAIDA_ROC, index=False)

    print(f"\nmetricas salvas em {SAIDA_MD.relative_to(RAIZ_CODIGO.parent).as_posix()}")
    print(f"curva ROC salva em {SAIDA_ROC.relative_to(RAIZ_CODIGO.parent).as_posix()}")

    diferenca = r_breast["auroc"] - r_busbra["auroc"]
    print(
        f"\n  AUROC BUS-BRA (interno) = {fmt(r_busbra['auroc'])} | "
        f"AUROC BrEaST (externo) = {fmt(r_breast['auroc'])} | "
        f"diferenca = {fmt(diferenca)}"
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
