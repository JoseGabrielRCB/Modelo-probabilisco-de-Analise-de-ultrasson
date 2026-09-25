"""Etapa 2 (BrEaST) do Protocolo C: portao de auditoria do indice do BrEaST, analogo a
`comum/verificar.py`.

Le `dados_processados/breast/indice_breast.csv` e, so para leitura, o `indice.csv` do BUS-BRA.

Travam a execucao se falharem:
    (a) nenhum sha1 duplicado dentro do BrEaST;
    (b) toda mascara existe e bate em dimensao com a imagem real (arquivo contra arquivo);
    (c) nenhum sha1 do BrEaST aparece no BUS-BRA — uma mesma imagem nas duas bases
        invalidaria a validacao externa.
So reporta:
    (d) balanco de classes do BrEaST.

Uso:
    python src/protocolo_c/verificar_breast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from comum.indexar import exigir

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA = RAIZ_CODIGO / "dados_processados" / "breast" / "indice_breast.csv"
ENTRADA_BUSBRA = RAIZ_CODIGO / "dados_processados" / "indice.csv"  # so leitura


def verificar_sha1_sem_duplicata(tabela: pd.DataFrame) -> None:
    """(a) Nenhuma imagem do BrEaST com o mesmo conteudo (sha1) que outra do BrEaST."""
    duplicados = tabela["sha1"].duplicated(keep=False)
    exigir(
        not duplicados.any(),
        f"(a) FALHOU: {int(duplicados.sum())} imagem(ns) do BrEaST com sha1 repetido; "
        "primeiros casos:\n  "
        + "\n  ".join(tabela.loc[duplicados, "imagem"].head(10)),
    )
    print(f"(a) OK: {len(tabela)} imagens, nenhum sha1 duplicado")


def verificar_mascaras(tabela: pd.DataFrame) -> None:
    """(b) Mascara existe e tem as mesmas dimensoes da imagem (ambas lidas com PIL)."""
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
    print(f"(b) OK: {len(tabela)} mascaras com a mesma dimensao da imagem")


def verificar_sem_intersecao_com_busbra(tabela_breast: pd.DataFrame) -> None:
    """(c) Nenhum sha1 do BrEaST no BUS-BRA. Sem o indice do BUS-BRA, avisa e pula."""
    if not ENTRADA_BUSBRA.exists():
        print(
            f"(c) PULADA: {ENTRADA_BUSBRA} nao encontrado (rodar indexar.py)"
        )
        return

    sha1_busbra = set(pd.read_csv(ENTRADA_BUSBRA, usecols=["sha1"])["sha1"])
    sha1_breast = set(tabela_breast["sha1"])
    intersecao = sha1_breast & sha1_busbra

    exigir(
        not intersecao,
        f"(c) FALHOU: {len(intersecao)} imagem(ns) com sha1 idêntico aparecem TANTO no "
        "BrEaST quanto no BUS-BRA — isso invalidaria o Protocolo C como validação "
        "externa (uma mesma imagem contaria como treino e teste); primeiros hashes:\n  "
        + "\n  ".join(sorted(intersecao)[:10]),
    )
    print(f"(c) OK: nenhuma das {len(sha1_breast)} imagens do BrEaST aparece nas "
          f"{len(sha1_busbra)} do BUS-BRA")


def reportar_balanco(tabela: pd.DataFrame) -> None:
    """(d) So reporta: balanco de classes do BrEaST inteiro."""
    malignas = int(tabela["rotulo"].sum())
    proporcao = f"{100 * malignas / len(tabela):.1f}".replace(".", ",")
    print(f"(d) classes (informativo): "
          f"{len(tabela)} imagens, {malignas} malignas ({proporcao}%)")


def main() -> None:
    exigir(ENTRADA.exists(), f"não encontrei {ENTRADA}; rode indexar_breast.py antes")
    tabela = pd.read_csv(ENTRADA, dtype={"birads": str})

    verificar_sha1_sem_duplicata(tabela)
    verificar_mascaras(tabela)
    verificar_sem_intersecao_com_busbra(tabela)
    reportar_balanco(tabela)

    print("\nverificacao OK: nenhuma falha")


if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        # Usar esse metodo é um pouco estranho , mas foi nescessario para poder validar um erro ocorrido e tambem por questao de docuemntao
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
