"""Etapa 6 do caminho ResNet18: calcula as métricas finais a partir de
`resultados/predicoes_resnet.csv`.

Cópia adaptada de `metricas.py` (caminho clássico) -- MESMA lógica (mesmo bootstrap por
paciente, mesmos limiares fora-de-fold, mesmo critério de sanidade), único ponto que
muda é a fonte (`predicoes_resnet.csv` em vez de `predicoes.csv`) e os arquivos de saída
(`metricas_resnet.md` / `roc_resnet.csv`, nunca `metricas.md` / `roc.csv`, para não
sobrescrever o resultado clássico já verificado). Ver `metricas.py` para a documentação
completa do método -- não repetida aqui em detalhe para não divergir das duas cópias com
o tempo; qualquer mudança de fundo no método deve ser espelhada manualmente nos dois
arquivos.

Para cada combinação de `protocolo` + `unidade` presente no arquivo, calcula AUROC e
AUPRC com IC95% por bootstrap (1.000 repetições, reamostrando pacientes), e no limiar
fixo 0,5 mais os limiares de Youden e sensibilidade>=0,90 (escolhidos fora de fold).

Escreve:
    - `resultados/metricas_resnet.md`
    - `resultados/roc_resnet.csv`

Uso:
    python src/resnet/metricas_resnet.py
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

# ---------------------------------------------------------------------------
# Caminhos -- tudo derivado da posição deste arquivo (ver indexar.py).
# ---------------------------------------------------------------------------

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo

ENTRADA = RAIZ_CODIGO / "resultados" / "predicoes_resnet.csv"
SAIDA_MD = RAIZ_CODIGO / "resultados" / "metricas_resnet.md"
SAIDA_ROC = RAIZ_CODIGO / "resultados" / "roc_resnet.csv"

SEMENTE = 42
N_BOOTSTRAP = 1000
SENSIBILIDADE_MINIMA = 0.90
LIMIAR_AVISO_ACURACIA = 0.95  # ver "critério de sanidade" no docstring de metricas.py


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com uma mensagem clara se a condição não valer."""
    if not condicao:
        raise ValueError(mensagem)


def ler_predicoes() -> pd.DataFrame:
    exigir(ENTRADA.exists(), f"não encontrei {ENTRADA}; rode experimento_resnet.py antes")
    tabela = pd.read_csv(ENTRADA)
    colunas_esperadas = {
        "protocolo", "unidade", "dataset", "id", "paciente", "fold", "y_true", "y_score",
    }
    exigir(
        colunas_esperadas.issubset(tabela.columns),
        f"faltam colunas em {ENTRADA.name}: {colunas_esperadas - set(tabela.columns)}",
    )
    return tabela


# ---------------------------------------------------------------------------
# Limiares
# ---------------------------------------------------------------------------

def limiar_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Limiar que maximiza sensibilidade + especificidade - 1 (índice J de Youden)."""
    fpr, tpr, limiares = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(limiares[np.argmax(j)])


def maior_limiar_sensibilidade_minima(
    y_true: np.ndarray, y_score: np.ndarray, sensibilidade_minima: float = SENSIBILIDADE_MINIMA
) -> float:
    """O maior limiar (mais específico) que ainda garante sensibilidade >= mínima."""
    fpr, tpr, limiares = roc_curve(y_true, y_score)
    validos = tpr >= sensibilidade_minima
    exigir(
        validos.any(),
        f"nenhum limiar atinge sensibilidade >= {sensibilidade_minima} nesta amostra",
    )
    return float(limiares[validos].max())


def limiares_fora_de_fold(y_true: np.ndarray, y_score: np.ndarray, fold: np.ndarray):
    """Para cada fold, escolhe os limiares (Youden, sens>=0.90) usando só as predições
    dos OUTROS folds — ver "Limiares fora-de-fold" no docstring de metricas.py."""
    youden_por_fold: dict[int, float] = {}
    sens90_por_fold: dict[int, float] = {}
    for f in np.unique(fold):
        fora = fold != f
        youden_por_fold[int(f)] = limiar_youden(y_true[fora], y_score[fora])
        sens90_por_fold[int(f)] = maior_limiar_sensibilidade_minima(y_true[fora], y_score[fora])

    limiar_youden_linha = np.array([youden_por_fold[int(f)] for f in fold])
    limiar_sens90_linha = np.array([sens90_por_fold[int(f)] for f in fold])
    return limiar_youden_linha, limiar_sens90_linha, youden_por_fold, sens90_por_fold


# ---------------------------------------------------------------------------
# Bootstrap por paciente
# ---------------------------------------------------------------------------

def gerar_reamostragens_por_paciente(
    paciente: np.ndarray, n_reps: int, semente: int
) -> list[np.ndarray]:
    """Gera `n_reps` conjuntos de posições de linha, cada um obtido reamostrando
    PACIENTES com reposição (não linhas soltas) e pegando todas as linhas de cada
    paciente sorteado -- ver docstring de metricas.py para a justificativa completa."""
    indices_por_paciente = pd.Series(np.arange(len(paciente))).groupby(paciente).apply(
        lambda s: s.to_numpy()
    )
    pacientes_unicos = indices_por_paciente.index.to_numpy()

    rng = np.random.default_rng(semente)
    reamostragens = []
    for _ in range(n_reps):
        escolhidos = rng.choice(pacientes_unicos, size=len(pacientes_unicos), replace=True)
        posicoes = np.concatenate([indices_por_paciente[p] for p in escolhidos])
        reamostragens.append(posicoes)
    return reamostragens


def ic95_bootstrap(funcao_metrica, reamostragens: list[np.ndarray], *colunas: np.ndarray):
    """Aplica `funcao_metrica` a cada reamostragem e devolve o percentil [2,5%, 97,5%]."""
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


# ---------------------------------------------------------------------------
# Métricas de um combo (protocolo, unidade)
# ---------------------------------------------------------------------------

def metricas_de_um_combo(df: pd.DataFrame) -> dict:
    """Calcula todas as métricas de uma combinação protocolo+unidade já isolada em `df`."""
    df = df.reset_index(drop=True)
    y_true = df["y_true"].to_numpy()
    y_score = df["y_score"].to_numpy()
    fold = df["fold"].to_numpy()
    paciente = df["paciente"].to_numpy()

    limiar_youden_linha, limiar_sens90_linha, youden_por_fold, sens90_por_fold = (
        limiares_fora_de_fold(y_true, y_score, fold)
    )

    pontos = {
        "0.5": (y_score >= 0.5).astype(int),
        "youden": (y_score >= limiar_youden_linha).astype(int),
        f"sens{int(SENSIBILIDADE_MINIMA * 100)}": (y_score >= limiar_sens90_linha).astype(int),
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
            "sensibilidade": float(sensibilidade),
            "sensibilidade_ic": sens_ic,
            "especificidade": float(especificidade),
            "especificidade_ic": espec_ic,
            "acuracia": float(acuracia),
            "acuracia_ic": acc_ic,
            "tn": int(matriz[0, 0]), "fp": int(matriz[0, 1]),
            "fn": int(matriz[1, 0]), "tp": int(matriz[1, 1]),
            "fpr": float(1 - especificidade),
            "tpr": float(sensibilidade),
        }

    resultado["limiares_por_fold"] = {
        "youden": youden_por_fold,
        f"sens{int(SENSIBILIDADE_MINIMA * 100)}": sens90_por_fold,
    }

    return resultado


# ---------------------------------------------------------------------------
# Formatação / saída
# ---------------------------------------------------------------------------

def fmt(valor: float, casas: int = 3) -> str:
    """Formata um número com vírgula decimal (padrão brasileiro do texto do TCC)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def fmt_ic(pontual: float, ic: tuple[float, float], casas: int = 3) -> str:
    return f"{fmt(pontual, casas)} [{fmt(ic[0], casas)}, {fmt(ic[1], casas)}]"


def montar_markdown(resultados: dict[tuple[str, str], dict]) -> str:
    linhas = ["# Métricas do experimento — caminho ResNet18 (comparação)", ""]
    linhas.append(
        "Características extraídas por `features_resnet.py` (ResNet18 pré-treinada em "
        "ImageNet, penúltima camada, 512-d) em vez das características clássicas de "
        "`features.py` -- ver `resultados/metricas.md` para o resultado clássico "
        "(scikit-image + regressão logística). MESMO classificador, MESMOS folds, MESMA "
        "semente nos dois caminhos; só a extração de características muda."
    )
    linhas.append("")
    linhas.append(
        f"IC95% por bootstrap (n={N_BOOTSTRAP}), reamostrando pacientes — ver "
        "`metricas_resnet.py` (cópia adaptada de `metricas.py`) para o método completo."
    )
    linhas.append("")

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

        linhas.append("Pontos de operação (limiares Youden e sensibilidade>=0,90 escolhidos "
                       "fora de fold — ver docstring de metricas.py):")
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
    linhas.append(
        "Bloco de dados de máquina (não editar à mão; mesmo formato de `metricas.py`, "
        "reservado para uso futuro por um `relatorio_resnet.py`, que não existe ainda):"
    )
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
    """Pontos da curva ROC completa (não bootstrap) de cada combo."""
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
    """Não trava — só avisa bem alto se a acurácia do Protocolo A por paciente (limiar
    0,5) passar de LIMIAR_AVISO_ACURACIA -- ver docstring de metricas.py."""
    chave = ("A", "paciente")
    if chave not in resultados:
        return
    acuracia = resultados[chave]["pontos_operacao"]["0.5"]["acuracia"]
    if acuracia > LIMIAR_AVISO_ACURACIA:
        aviso = (
            f"acurácia do Protocolo A por paciente = {fmt(acuracia)}, acima de "
            f"{fmt(LIMIAR_AVISO_ACURACIA)}"
        )
        borda = "!" * 78
        print(f"\n{borda}", file=sys.stderr)
        print("AVISO: resultado bom demais para desconfiar por padrão.", file=sys.stderr)
        print(aviso, file=sys.stderr)
        print(
            "Isso É UM INDÍCIO de erro no particionamento (paciente cruzando fold, "
            "vazamento) — não é garantia de erro, mas confira antes de reportar este "
            "número como definitivo.",
            file=sys.stderr,
        )
        print(f"{borda}\n", file=sys.stderr)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)  # zero_division já tratado

    tabela = ler_predicoes()
    combos = sorted(tabela[["protocolo", "unidade"]].drop_duplicates().itertuples(index=False, name=None))

    print(f"combinações protocolo+unidade encontradas: {combos}")

    resultados: dict[tuple[str, str], dict] = {}
    for protocolo, unidade in combos:
        parte = tabela[(tabela["protocolo"] == protocolo) & (tabela["unidade"] == unidade)]
        print(f"calculando métricas de {protocolo}/{unidade} ({len(parte)} linhas)...")
        resultados[(protocolo, unidade)] = metricas_de_um_combo(parte)

    verificar_criterio_de_sanidade(resultados)

    SAIDA_MD.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_MD.write_text(montar_markdown(resultados), encoding="utf-8")

    roc = montar_roc_csv(tabela, combos)
    roc.to_csv(SAIDA_ROC, index=False)

    print(f"\nmétricas salvas em {SAIDA_MD.relative_to(RAIZ_CODIGO.parent).as_posix()}")
    print(f"curva ROC salva em {SAIDA_ROC.relative_to(RAIZ_CODIGO.parent).as_posix()}")

    for (protocolo, unidade), r in resultados.items():
        print(
            f"  {protocolo}/{unidade}: AUROC={fmt(r['auroc'])} "
            f"acurácia@0,5={fmt(r['pontos_operacao']['0.5']['acuracia'])}"
        )


if __name__ == "__main__":
    # O console do Windows costuma abrir em cp1252 e comeria os acentos das mensagens.
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
