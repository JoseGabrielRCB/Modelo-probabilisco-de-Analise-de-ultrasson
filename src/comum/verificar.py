"""Etapa 3 do pipeline: portão de auditoria — confere `indice_particionado.csv` contra os
arquivos de origem, em vez de confiar cegamente neles.

Este é o script standalone que sustenta o argumento central de rigor metodológico do TCC:
comparar o que o índice diz com o que está de fato no disco, e travar com um erro claro se
algo não bater. Pode ser executado a qualquer momento, sozinho, sem precisar rodar os
outros scripts do pipeline antes (só precisa que `indice_particionado.csv` já exista) e
sem precisar de GPU.

Verificações que travam a execução se falharem:
    (a) nenhum paciente aparece em mais de um fold;
    (b) nenhum sha1 duplicado;
    (c) toda imagem tem uma máscara correspondente no disco, e as dimensões da máscara
        batem com as dimensões da IMAGEM — ambas lidas do arquivo real com PIL, nunca
        da metadata declarada em `largura`/`altura` (ver nota de auditoria em
        `verificar_mascaras`, achado real em bus_data.csv).

Verificação que só reporta (não trava):
    (d) balanço de classes (proporção de malignas) por fold.

Uso:
    python src/comum/verificar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Caminhos — tudo derivado da posição deste arquivo (ver indexar.py).
# ---------------------------------------------------------------------------

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA = RAIZ_CODIGO / "dados_processados" / "indice_particionado.csv"


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com uma mensagem clara se a condição não valer.

    Mesma função de `indexar.py`/`particionar.py`: parar com erro, nunca seguir em
    frente com dado suspeito.
    """
    if not condicao:
        raise ValueError(mensagem)


def ler_indice_particionado() -> pd.DataFrame:
    exigir(
        ENTRADA.exists(),
        f"não encontrei {ENTRADA}; rode indexar.py e particionar.py antes",
    )
    return pd.read_csv(ENTRADA, dtype={"birads": str})


def verificar_paciente_unico_por_fold(tabela: pd.DataFrame) -> None:
    """(a) Nenhum paciente pode ter imagens em mais de um fold.

    Repete o "gate" já verificado em `particionar.py` — de propósito: este script
    existe justamente para não depender de que a verificação de outro script tenha
    corrido, ou tenha corrido corretamente. É a mesma pergunta, feita de novo, de forma
    independente.
    """
    folds_por_paciente = tabela.groupby("paciente")["fold"].nunique()
    ruins = folds_por_paciente[folds_por_paciente > 1]
    exigir(
        ruins.empty,
        f"(a) FALHOU: {len(ruins)} paciente(s) com imagens em mais de um fold; "
        "primeiros casos:\n  " + "\n  ".join(str(p) for p in ruins.head(10).index),
    )
    print(f"(a) OK — nenhum dos {tabela['paciente'].nunique()} pacientes cruza fold.")


def verificar_sha1_sem_duplicata(tabela: pd.DataFrame) -> None:
    """(b) Nenhuma imagem pode ter o mesmo conteúdo (sha1) que outra."""
    duplicados = tabela["sha1"].duplicated(keep=False)
    exigir(
        not duplicados.any(),
        f"(b) FALHOU: {int(duplicados.sum())} imagem(ns) com sha1 repetido; "
        "primeiros casos:\n  "
        + "\n  ".join(tabela.loc[duplicados, "imagem"].head(10)),
    )
    print(f"(b) OK — nenhum sha1 duplicado entre as {len(tabela)} imagens.")


def verificar_mascaras(tabela: pd.DataFrame) -> None:
    """(c) Toda imagem tem uma máscara correspondente no disco, e as dimensões da
    máscara batem com as dimensões da IMAGEM — as duas lidas do arquivo real com PIL,
    nunca contra as colunas `largura`/`altura` do índice (que só refletem o que
    `bus_data.csv` declarou, e podem estar erradas — ver nota de auditoria abaixo).

    Não basta a máscara existir: se as dimensões não baterem, a máscara pode ter vindo
    trocada, ou redimensionada por engano em algum passo anterior — um erro silencioso
    que só aparece se alguém de fato abrir os dois arquivos e medir.

    Nota de auditoria (achado real, não hipotético): a primeira versão deste check
    comparava a máscara contra `largura`/`altura` do índice (ou seja, contra a
    metadata declarada em `bus_data.csv`) e travava para 4 das 1.875 linhas —
    bus_0856-l, bus_0857-r, bus_0912-l e bus_0990-l (0,2% dos casos). Investigando,
    confirmou-se que o campo `Height` de `bus_data.csv` está ERRADO nessas 4 linhas
    (não bate com o arquivo de imagem real no disco), mas a imagem real e a máscara
    real CONCORDAM perfeitamente entre si nos 4 casos — não há desalinhamento nenhum
    entre imagem e máscara, só um erro de metadata no CSV de origem. Por isso o check
    foi corrigido para comparar máscara-real contra imagem-real (nunca usando a
    metadata do índice como referência para nenhum dos dois lados) — mais robusto de
    qualquer forma, porque deixa de depender de a metadata do dataset estar certa.
    """
    problemas: list[str] = []
    for linha in tabela.itertuples(index=False):
        caminho_imagem = RAIZ_TCC / linha.imagem
        caminho_mascara = RAIZ_TCC / linha.mascara

        if not caminho_imagem.exists():
            problemas.append(f"{linha.id}: imagem não existe em {linha.imagem}")
            continue
        if not caminho_mascara.exists():
            problemas.append(f"{linha.id}: máscara não existe em {linha.mascara}")
            continue

        with Image.open(caminho_imagem) as img:
            tamanho_imagem = img.size
        with Image.open(caminho_mascara) as msk:
            tamanho_mascara = msk.size

        if tamanho_mascara != tamanho_imagem:
            problemas.append(
                f"{linha.id}: máscara {tamanho_mascara[0]}x{tamanho_mascara[1]} != "
                f"imagem {tamanho_imagem[0]}x{tamanho_imagem[1]} (ambas lidas do arquivo real)"
            )

    exigir(
        not problemas,
        f"(c) FALHOU: {len(problemas)} problema(s) de máscara; primeiros casos:\n  "
        + "\n  ".join(problemas[:10]),
    )
    print(f"(c) OK — todas as {len(tabela)} máscaras batem em dimensão com a imagem real "
          "(comparação arquivo-a-arquivo, não contra metadata declarada).")


def reportar_balanco_por_fold(tabela: pd.DataFrame) -> None:
    """(d) Só reporta, não trava: balanço de classes (proporção de malignas) por fold.

    Um fold muito desbalanceado em relação aos outros não é necessariamente um erro
    (os folds são os oficiais do dataset, não escolhidos por nós), mas é informação
    relevante para interpretar os resultados por fold no Protocolo A.
    """
    print("(d) balanço de classes por fold (informativo, não é critério de falha):")
    for fold, parte in tabela.groupby("fold", sort=True):
        malignas = int(parte["rotulo"].sum())
        proporcao = f"{100 * malignas / len(parte):.1f}".replace(".", ",")
        print(
            f"    fold {fold}: {len(parte)} imagens, {malignas} malignas ({proporcao}%)"
        )


def main() -> None:
    tabela = ler_indice_particionado()

    verificar_paciente_unico_por_fold(tabela)
    verificar_sha1_sem_duplicata(tabela)
    verificar_mascaras(tabela)
    reportar_balanco_por_fold(tabela)

    print("\nverificação completa: todos os portões (a)-(c) passaram.")


if __name__ == "__main__":
    # O console do Windows costuma abrir em cp1252 e comeria os acentos das mensagens.
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
