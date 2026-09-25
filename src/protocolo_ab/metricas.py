"""Etapa 6 do pipeline: calcula as metricas finais a partir de `resultados/predicoes.csv`.

Para cada combinacao protocolo + unidade (A/imagem, A/paciente, B/imagem):
    - AUROC e AUPRC, com IC95% por bootstrap (1.000 repeticoes, reamostrando PACIENTES);
    - em 3 pontos de operacao (limiar fixo 0,5, Youden e maior limiar com sensibilidade
      >= 0,90): sensibilidade, especificidade, acuracia e matriz de confusao, com IC95%.

Limiares fora-de-fold: o limiar (Youden / sens>=0,90) aplicado as linhas do fold f e
escolhido so com as predicoes dos OUTROS folds. Nenhuma linha e decidida por um limiar que
a tenha usado; por isso o limiar varia levemente de fold para fold.

Escreve `resultados/metricas.md` e `resultados/roc.csv`. O `.md` termina com um bloco JSON
em comentario HTML (DADOS_MAQUINA) com os pontos (fpr, tpr), lido por `relatorio.py`.

Alerta (nao trava): acuracia do Protocolo A/paciente@0,5 acima de 0,95 indica provavel erro
de particionamento.

`resnet/metricas_resnet.py` e `protocolo_c/metricas_breast.py` reaproveitam este modulo.

Uso:
    python src/protocolo_ab/metricas.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from comum.indexar import exigir

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo

ENTRADA = RAIZ_CODIGO / "resultados" / "predicoes.csv"
SAIDA_MD = RAIZ_CODIGO / "resultados" / "metricas.md"
SAIDA_ROC = RAIZ_CODIGO / "resultados" / "roc.csv"

SEMENTE = 42
N_BOOTSTRAP = 1000
SENSIBILIDADE_MINIMA = 0.90
LIMIAR_AVISO_ACURACIA = 0.95

# Textos fixos do metricas.md (a ResNet passa os seus, ver resnet/metricas_resnet.py)
TEXTOS_MD = {
    "cabecalho": [
        "# Métricas do experimento",
        "",
        f"IC95% por bootstrap (n={N_BOOTSTRAP}), reamostrando pacientes — ver "
        "`metricas.py` para o método completo.",
        "",
    ],
    "pontos": "Pontos de operação (limiares Youden e sensibilidade>=0,90 escolhidos "
              "fora de fold — ver docstring do módulo):",
    "maquina": "Bloco de dados de máquina para `relatorio.py` (não editar à mão; ver docstring "
               "de `metricas.py`, seção \"Bloco de dados de máquina\"):",
}


# Limiares

def limiar_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Limiar que maximiza sensibilidade + especificidade - 1 (indice J de Youden)."""
    fpr, tpr, limiares = roc_curve(y_true, y_score)
    return float(limiares[np.argmax(tpr - fpr)])


def maior_limiar_sensibilidade_minima(
    y_true: np.ndarray, y_score: np.ndarray, sensibilidade_minima: float = SENSIBILIDADE_MINIMA
) -> float:
    """Maior limiar (mais especifico) que ainda garante sensibilidade >= minima."""
    fpr, tpr, limiares = roc_curve(y_true, y_score)
    validos = tpr >= sensibilidade_minima
    exigir(
        validos.any(),
        f"nenhum limiar atinge sensibilidade >= {sensibilidade_minima} nesta amostra",
    )
    return float(limiares[validos].max())


def limiares_fora_de_fold(y_true: np.ndarray, y_score: np.ndarray, fold: np.ndarray):
    """Limiares (Youden, sens>=0,90) de cada linha, escolhidos so com os OUTROS folds."""
    youden_por_fold: dict[int, float] = {}
    sens90_por_fold: dict[int, float] = {}
    for f in np.unique(fold):
        fora = fold != f
        youden_por_fold[int(f)] = limiar_youden(y_true[fora], y_score[fora])
        sens90_por_fold[int(f)] = maior_limiar_sensibilidade_minima(y_true[fora], y_score[fora])

    limiar_youden_linha = np.array([youden_por_fold[int(f)] for f in fold])
    limiar_sens90_linha = np.array([sens90_por_fold[int(f)] for f in fold])
    return limiar_youden_linha, limiar_sens90_linha


# Bootstrap por paciente

def gerar_reamostragens_por_paciente(
    paciente: np.ndarray, n_reps: int, semente: int
) -> list[np.ndarray]:
    """Sorteia PACIENTES com reposicao n_reps vezes (imagens do paciente nao sao independentes)."""
    indices_por_paciente = pd.Series(np.arange(len(paciente))).groupby(paciente).apply(
        lambda s: s.to_numpy()
    )
    pacientes_unicos = indices_por_paciente.index.to_numpy()

    rng = np.random.default_rng(semente)
    reamostragens = []
    for _ in range(n_reps):
        escolhidos = rng.choice(pacientes_unicos, size=len(pacientes_unicos), replace=True)
        reamostragens.append(np.concatenate([indices_por_paciente[p] for p in escolhidos]))
    return reamostragens


def ic95_bootstrap(funcao_metrica, reamostragens: list[np.ndarray], *colunas: np.ndarray):
    """Percentis 2,5 e 97,5; descarta amostras de uma so classe e trava se descartar mais de 10%."""
    valores = []
    for posicoes in reamostragens:
        argumentos = [coluna[posicoes] for coluna in colunas]
        try:
            valores.append(funcao_metrica(*argumentos))
        except ValueError:
            continue
    exigir(
        len(valores) >= 0.9 * len(reamostragens),
        f"bootstrap descartou {len(reamostragens) - len(valores)}/{len(reamostragens)} "
        "repetições (amostra sem as duas classes) — resultado poucos confiável",
    )
    return tuple(np.percentile(valores, [2.5, 97.5]))


# Metricas

def metricas_nos_pontos(y_true: np.ndarray, y_score: np.ndarray, paciente: np.ndarray,
                        pontos: dict[str, np.ndarray]) -> dict:
    """AUROC/AUPRC e, por ponto de operacao: sensibilidade, especificidade, acuracia e matriz, com IC95%."""
    reamostragens = gerar_reamostragens_por_paciente(paciente, N_BOOTSTRAP, SEMENTE)

    resultado: dict = {"n_linhas": len(y_true), "n_pacientes": len(np.unique(paciente))}

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
            # ponto (fpr, tpr) para marcar na curva ROC
            "fpr": float(1 - especificidade), "tpr": float(sensibilidade),
        }

    return resultado


def metricas_de_um_combo(df: pd.DataFrame) -> dict:
    """Metricas de um combo protocolo+unidade, com limiares escolhidos fora de fold."""
    df = df.reset_index(drop=True)
    y_true = df["y_true"].to_numpy()
    y_score = df["y_score"].to_numpy()
    limiar_youden_linha, limiar_sens90_linha = limiares_fora_de_fold(
        y_true, y_score, df["fold"].to_numpy()
    )
    pontos = {
        "0.5": (y_score >= 0.5).astype(int),
        "youden": (y_score >= limiar_youden_linha).astype(int),
        f"sens{int(SENSIBILIDADE_MINIMA * 100)}": (y_score >= limiar_sens90_linha).astype(int),
    }
    return metricas_nos_pontos(y_true, y_score, df["paciente"].to_numpy(), pontos)


# Formatacao e saida

def fmt(valor: float, casas: int = 3) -> str:
    """Numero com virgula decimal (padrao do texto do TCC)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def fmt_ic(pontual: float, ic: tuple[float, float], casas: int = 3) -> str:
    return f"{fmt(pontual, casas)} [{fmt(ic[0], casas)}, {fmt(ic[1], casas)}]"


def montar_markdown(resultados: dict[tuple[str, str], dict], textos: dict = TEXTOS_MD) -> str:
    linhas = list(textos["cabecalho"])

    for (protocolo, unidade), r in resultados.items():
        linhas.append(f"## Protocolo {protocolo} — unidade: {unidade}")
        linhas.append("")
        linhas.append(f"{r['n_linhas']} linhas, {r['n_pacientes']} pacientes distintos.")
        linhas.append("")
        linhas.append("| métrica | valor | IC95% |")
        linhas.append("|---|---|---|")
        linhas.append(f"| AUROC | {fmt(r['auroc'])} | [{fmt(r['auroc_ic'][0])}, {fmt(r['auroc_ic'][1])}] |")
        linhas.append(f"| AUPRC | {fmt(r['auprc'])} | [{fmt(r['auprc_ic'][0])}, {fmt(r['auprc_ic'][1])}] |")
        linhas.append("")

        linhas.append(textos["pontos"])
        linhas.append("")
        linhas.append("| ponto | sensibilidade | especificidade | acurácia | TP | TN | FP | FN |")
        linhas.append("|---|---|---|---|---|---|---|---|")
        nomes_legiveis = {"0.5": "limiar 0,5 (fixo)", "youden": "limiar de Youden",
                          f"sens{int(SENSIBILIDADE_MINIMA*100)}": f"sensibilidade >= {fmt(SENSIBILIDADE_MINIMA, 2)}"}
        for nome, p in r["pontos_operacao"].items():
            linhas.append(
                f"| {nomes_legiveis.get(nome, nome)} "
                f"| {fmt_ic(p['sensibilidade'], p['sensibilidade_ic'])} "
                f"| {fmt_ic(p['especificidade'], p['especificidade_ic'])} "
                f"| {fmt_ic(p['acuracia'], p['acuracia_ic'])} "
                f"| {p['tp']} | {p['tn']} | {p['fp']} | {p['fn']} |"
            )
        linhas.append("")

    linhas.append("---")
    linhas.append("")
    linhas.append(textos["maquina"])
    linhas.append("")
    dados_maquina = {
        f"{protocolo}|{unidade}": {
            nome: {"fpr": p["fpr"], "tpr": p["tpr"]}
            for nome, p in r["pontos_operacao"].items()
        }
        for (protocolo, unidade), r in resultados.items()
    }
    linhas.append(f"<!-- DADOS_MAQUINA: {json.dumps(dados_maquina)} -->")
    linhas.append("")

    return "\n".join(linhas)


def montar_roc_csv(tabela: pd.DataFrame, combos: list[tuple[str, str]]) -> pd.DataFrame:
    """Pontos da curva ROC completa (sem bootstrap) de cada combo."""
    partes = []
    for protocolo, unidade in combos:
        parte = tabela[(tabela["protocolo"] == protocolo) & (tabela["unidade"] == unidade)]
        fpr, tpr, limiares = roc_curve(parte["y_true"], parte["y_score"])
        partes.append(pd.DataFrame({
            "protocolo": protocolo, "unidade": unidade,
            "fpr": fpr, "tpr": tpr, "limiar": limiares,
        }))
    return pd.concat(partes, ignore_index=True)


def verificar_criterio_de_sanidade(resultados: dict[tuple[str, str], dict]) -> None:
    """So avisa (nao trava) se a acuracia A/paciente@0,5 passar de LIMIAR_AVISO_ACURACIA."""
    chave = ("A", "paciente")
    if chave not in resultados:
        return
    acuracia = resultados[chave]["pontos_operacao"]["0.5"]["acuracia"]
    if acuracia > LIMIAR_AVISO_ACURACIA:
        borda = "---------"
        print(f"\n{borda}", file=sys.stderr)
        print("AVISO: acuracia alta demais", file=sys.stderr)
        print(f"Protocolo A/paciente: acuracia {fmt(acuracia)} (limite "
              f"{fmt(LIMIAR_AVISO_ACURACIA)})", file=sys.stderr)
        print("Possivel vazamento entre folds. Conferir particionamento antes de reportar.",
              file=sys.stderr)
        print(f"{borda}\n", file=sys.stderr)


def main(entrada: Path = ENTRADA, saida_md: Path = SAIDA_MD, saida_roc: Path = SAIDA_ROC,
         textos: dict = TEXTOS_MD) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)  # zero_division ja tratado

    tabela = pd.read_csv(entrada)
    combos = sorted(tabela[["protocolo", "unidade"]].drop_duplicates().itertuples(index=False, name=None))

    resultados: dict[tuple[str, str], dict] = {}
    for protocolo, unidade in combos:
        parte = tabela[(tabela["protocolo"] == protocolo) & (tabela["unidade"] == unidade)]
        resultados[(protocolo, unidade)] = metricas_de_um_combo(parte)

    verificar_criterio_de_sanidade(resultados)

    saida_md.parent.mkdir(parents=True, exist_ok=True)
    saida_md.write_text(montar_markdown(resultados, textos), encoding="utf-8")
    montar_roc_csv(tabela, combos).to_csv(saida_roc, index=False)

    print(f"\nmetricas salvas em {saida_md.relative_to(RAIZ_CODIGO.parent).as_posix()}")
    print(f"curva ROC salva em {saida_roc.relative_to(RAIZ_CODIGO.parent).as_posix()}")
    for (protocolo, unidade), r in resultados.items():
        print(
            f"  {protocolo}/{unidade}: AUROC={fmt(r['auroc'])} "
            f"acuracia@0,5={fmt(r['pontos_operacao']['0.5']['acuracia'])}"
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
