"""Etapa 2 (BrEaST) do Protocolo C: portão de auditoria do índice do BrEaST — mesmo
espírito de `verificar.py` (o portão do BUS-BRA), adaptado às particularidades do BrEaST.

Lê só `dados_processados/breast/indice_breast.csv` (saída de `indexar_breast.py`) e,
opcionalmente, `dados_processados/indice.csv` do BUS-BRA — em modo SOMENTE LEITURA, só
para checar que nenhuma imagem se repete entre as duas bases (checagem (c) abaixo). Este
script nunca escreve nada em `dados_processados/indice.csv` nem em qualquer arquivo do
BUS-BRA.

Verificações que travam a execução se falharem:
    (a) nenhum sha1 duplicado dentro do BrEaST;
    (b) toda máscara bate em dimensão com a imagem real, arquivo a arquivo (mesma técnica
        corrigida em `verificar.py`: nunca contra metadata declarada — aqui nem existe
        metadata declarada de largura/altura na planilha do BrEaST, então a comparação já
        nasce arquivo-a-arquivo);
    (c) nenhum sha1 do BrEaST aparece também no BUS-BRA — evita que uma imagem
        "vazasse" para as duas bases ao mesmo tempo, o que invalidaria a comparação
        interno x externo do Protocolo C.

Verificação que só reporta (não trava):
    (d) balanço de classes do BrEaST.

Uso:
    python src/verificar_breast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

RAIZ_CODIGO = Path(__file__).resolve().parents[1]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA = RAIZ_CODIGO / "dados_processados" / "breast" / "indice_breast.csv"
# Só leitura, nunca escrita — ver docstring do módulo.
ENTRADA_BUSBRA = RAIZ_CODIGO / "dados_processados" / "indice.csv"


def exigir(condicao: bool, mensagem: str) -> None:
    """Mesma função das outras etapas: parar com erro, nunca seguir em frente com dado
    suspeito."""
    if not condicao:
        raise ValueError(mensagem)


def ler_indice_breast() -> pd.DataFrame:
    exigir(
        ENTRADA.exists(),
        f"não encontrei {ENTRADA}; rode indexar_breast.py antes",
    )
    return pd.read_csv(ENTRADA, dtype={"birads": str})


def verificar_sha1_sem_duplicata(tabela: pd.DataFrame) -> None:
    """(a) Nenhuma imagem do BrEaST pode ter o mesmo conteúdo (sha1) que outra do
    próprio BrEaST."""
    duplicados = tabela["sha1"].duplicated(keep=False)
    exigir(
        not duplicados.any(),
        f"(a) FALHOU: {int(duplicados.sum())} imagem(ns) do BrEaST com sha1 repetido; "
        "primeiros casos:\n  "
        + "\n  ".join(tabela.loc[duplicados, "imagem"].head(10)),
    )
    print(f"(a) OK — nenhum sha1 duplicado entre as {len(tabela)} imagens do BrEaST.")


def verificar_mascaras(tabela: pd.DataFrame) -> None:
    """(b) Toda imagem tem uma máscara correspondente no disco, e as dimensões da
    máscara batem com as dimensões da IMAGEM — as duas lidas do arquivo real com PIL
    (mesma técnica de `verificar.py`, ver nota de auditoria lá; aqui nem há metadata
    declarada de largura/altura na planilha para comparar por engano)."""
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
        f"(b) FALHOU: {len(problemas)} problema(s) de máscara; primeiros casos:\n  "
        + "\n  ".join(problemas[:10]),
    )
    print(f"(b) OK — todas as {len(tabela)} máscaras batem em dimensão com a imagem real "
          "(comparação arquivo-a-arquivo).")


def verificar_sem_intersecao_com_busbra(tabela_breast: pd.DataFrame) -> None:
    """(c) Nenhum sha1 do BrEaST pode aparecer também no BUS-BRA — SÓ LEITURA de
    `indice.csv`, nunca escrita (ver docstring do módulo). Se essa checagem não puder ser
    feita porque o índice do BUS-BRA ainda não existe, avisa e segue (não é um requisito
    de `indexar_breast.py`/`verificar_breast.py` sozinhos, só do Protocolo C completo)."""
    if not ENTRADA_BUSBRA.exists():
        print(
            f"(c) PULADA — {ENTRADA_BUSBRA} não existe ainda (rode indexar.py do BUS-BRA "
            "para habilitar esta checagem); não é um erro deste script."
        )
        return

    busbra = pd.read_csv(ENTRADA_BUSBRA, usecols=["sha1"])
    sha1_busbra = set(busbra["sha1"])
    sha1_breast = set(tabela_breast["sha1"])
    intersecao = sha1_breast & sha1_busbra

    exigir(
        not intersecao,
        f"(c) FALHOU: {len(intersecao)} imagem(ns) com sha1 idêntico aparecem TANTO no "
        "BrEaST quanto no BUS-BRA — isso invalidaria o Protocolo C como validação "
        "externa (uma mesma imagem contaria como treino e teste); primeiros hashes:\n  "
        + "\n  ".join(sorted(intersecao)[:10]),
    )
    print(f"(c) OK — nenhum dos {len(sha1_breast)} sha1 do BrEaST aparece nos "
          f"{len(sha1_busbra)} sha1 do BUS-BRA (bases sem sobreposição de imagens).")


def reportar_balanco(tabela: pd.DataFrame) -> None:
    """(d) Só reporta, não trava: balanço de classes do BrEaST inteiro (não há folds
    aqui — o Protocolo C não faz validação cruzada no BrEaST)."""
    malignas = int(tabela["rotulo"].sum())
    proporcao = f"{100 * malignas / len(tabela):.1f}".replace(".", ",")
    print(f"(d) balanço de classes do BrEaST (informativo, não é critério de falha): "
          f"{len(tabela)} imagens, {malignas} malignas ({proporcao}%)")


def main() -> None:
    tabela = ler_indice_breast()

    verificar_sha1_sem_duplicata(tabela)
    verificar_mascaras(tabela)
    verificar_sem_intersecao_com_busbra(tabela)
    reportar_balanco(tabela)

    print("\nverificação completa: todos os portões (a)-(b) [e (c), quando aplicável] passaram.")


if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
