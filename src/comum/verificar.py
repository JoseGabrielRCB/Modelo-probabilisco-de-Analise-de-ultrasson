"""Etapa 3 do pipeline: portao de auditoria — confere `indice_particionado.csv` contra os
arquivos de origem, em vez de confiar cegamente neles.

Este e o script standalone que sustenta o argumento central de rigor metodologico do TCC:
comparar o que o indice diz com o que esta de fato no disco, e travar com um erro claro se
algo nao bater. Pode ser executado a qualquer momento, sozinho, sem precisar rodar os
outros scripts do pipeline antes (so precisa que `indice_particionado.csv` ja exista) e
sem precisar de GPU.

Verificacoes que travam a execucao se falharem:
    (a) nenhum paciente aparece em mais de um fold;
    (b) nenhum sha1 duplicado;
    (c) toda imagem tem uma mascara correspondente no disco, e as dimensoes da mascara
        batem com as dimensoes da IMAGEM — ambas lidas do arquivo real com PIL, nunca
        da metadata declarada em `largura`/`altura` (ver nota de auditoria em
        `verificar_mascaras`, achado real em bus_data.csv).

Verificacao que so reporta (nao trava):
    (d) balanco de classes (proporcao de malignas) por fold.

Uso:
    python src/comum/verificar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

# Caminhos: derivados da posicao deste arquivo (ver indexar.py)

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA = RAIZ_CODIGO / "dados_processados" / "indice_particionado.csv"


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com mensagem clara se a condicao nao valer."""
    if not condicao:
        raise ValueError(mensagem)


def ler_indice_particionado() -> pd.DataFrame:
    exigir(
        ENTRADA.exists(),
        f"não encontrei {ENTRADA}; rode indexar.py e particionar.py antes",
    )
    return pd.read_csv(ENTRADA, dtype={"birads": str})


def verificar_paciente_unico_por_fold(tabela: pd.DataFrame) -> None:
    """(a) Nenhum paciente em mais de um fold; repete o gate do particionar.py de forma independente."""
    folds_por_paciente = tabela.groupby("paciente")["fold"].nunique()
    ruins = folds_por_paciente[folds_por_paciente > 1]
    exigir(
        ruins.empty,
        f"(a) FALHOU: {len(ruins)} paciente(s) com imagens em mais de um fold; "
        "primeiros casos:\n  " + "\n  ".join(str(p) for p in ruins.head(10).index),
    )
    print(f"(a) OK: {tabela['paciente'].nunique()} pacientes, nenhum cruza fold")


def verificar_sha1_sem_duplicata(tabela: pd.DataFrame) -> None:
    """(b) Nenhuma imagem pode ter o mesmo conteudo (sha1) que outra."""
    duplicados = tabela["sha1"].duplicated(keep=False)
    exigir(
        not duplicados.any(),
        f"(b) FALHOU: {int(duplicados.sum())} imagem(ns) com sha1 repetido; "
        "primeiros casos:\n  "
        + "\n  ".join(tabela.loc[duplicados, "imagem"].head(10)),
    )
    print(f"(b) OK: {len(tabela)} imagens, nenhum sha1 duplicado")


def verificar_mascaras(tabela: pd.DataFrame) -> None:
    """(c) Mascara existe e tem a dimensao da imagem, ambas lidas do arquivo (nao da metadata)."""
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
    print(f"(c) OK: {len(tabela)} mascaras com a mesma dimensao da imagem")


def reportar_balanco_por_fold(tabela: pd.DataFrame) -> None:
    """(d) So reporta (nao trava): proporcao de malignas por fold."""
    print("(d) classes por fold (informativo):")
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

    print("\nverificacao OK: (a), (b) e (c) passaram")


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
