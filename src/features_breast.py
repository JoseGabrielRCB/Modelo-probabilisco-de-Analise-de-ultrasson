"""Etapa 3 (BrEaST) do Protocolo C: calcula o MESMO vetor de características que
`features.py` calcula para o BUS-BRA, agora para o BrEaST.

Importa as funções de extração diretamente de `features.py` (`extrair_vetor`, que já
concatena as 5 categorias — intensidade, GLCM/Haralick, LBP, Sobel, grade 4x4 — na mesma
ordem, com o mesmo redimensionamento `TAMANHO` antes do cálculo) em vez de duplicar a
lógica. Isso é o que garante que a comparação BUS-BRA x BrEaST do Protocolo C mede de fato
o mesmo tipo de característica nas duas bases — duplicar a lógica aqui poderia divergir
silenciosamente de `features.py` e invalidar a comparação sem nenhum erro aparente.

Lê só `dados_processados/breast/indice_breast.csv` (nunca a planilha bruta do BrEaST
diretamente) e escreve, na pasta separada do BrEaST:
    - `dados_processados/breast/caracteristicas_breast.npy`      — matriz 252 x D
    - `dados_processados/breast/caracteristicas_breast_ids.csv`  — colunas id, paciente,
      rotulo (suficiente para o Protocolo C: não há fold no BrEaST)

Trava se D (o número de colunas da matriz do BrEaST) não bater EXATAMENTE com D do
BUS-BRA (`dados_processados/caracteristicas.npy`, lido aqui só para conferir a forma, não
como dado de treino) — os dois vetores precisam ter a mesma dimensão para o classificador
treinado no BUS-BRA (`experimento_c.py`) poder ser aplicado ao BrEaST sem erro.

Uso:
    python src/features_breast.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reaproveita a extração de características do BUS-BRA — mesmo diretório src/, ver
# docstring do módulo. `features.py` não é modificado por este import.
from features import TAMANHO, extrair_vetor
from features import verificar_sanidade as verificar_sanidade_matriz
from indexar import exigir

RAIZ_CODIGO = Path(__file__).resolve().parents[1]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA = RAIZ_CODIGO / "dados_processados" / "breast" / "indice_breast.csv"
# Só para conferir D — nunca usada como dado de treino aqui (ver docstring).
ENTRADA_MATRIZ_BUSBRA = RAIZ_CODIGO / "dados_processados" / "caracteristicas.npy"

SAIDA_MATRIZ = RAIZ_CODIGO / "dados_processados" / "breast" / "caracteristicas_breast.npy"
SAIDA_IDS = RAIZ_CODIGO / "dados_processados" / "breast" / "caracteristicas_breast_ids.csv"

COLUNAS_IDS = ["id", "paciente", "rotulo"]

LINHAS_ESPERADAS = 252


def ler_indice_breast() -> pd.DataFrame:
    exigir(
        ENTRADA.exists(),
        f"não encontrei {ENTRADA}; rode indexar_breast.py (e verificar_breast.py) antes",
    )
    return pd.read_csv(ENTRADA, dtype={"birads": str})


def extrair_todas(tabela: pd.DataFrame) -> np.ndarray:
    """Extrai o vetor de características de cada linha do índice do BrEaST, usando
    `extrair_vetor` de `features.py` (mesma função, mesmos parâmetros, mesma ordem de
    categorias que o BUS-BRA)."""
    vetores = []
    inicio = time.time()
    total = len(tabela)
    for posicao, linha in enumerate(tabela.itertuples(index=False), start=1):
        caminho = RAIZ_TCC / linha.imagem
        exigir(caminho.exists(), f"imagem não encontrada no disco: {caminho}")
        vetores.append(extrair_vetor(caminho))
        if posicao % 50 == 0 or posicao == total:
            decorrido = time.time() - inicio
            print(f"  {posicao}/{total} imagens processadas ({decorrido:.0f}s)")
    return np.vstack(vetores)


def verificar_dimensao_igual_busbra(matriz: np.ndarray) -> None:
    """Trava se D (número de colunas) da matriz do BrEaST não bater com D do BUS-BRA —
    ver docstring do módulo."""
    exigir(
        ENTRADA_MATRIZ_BUSBRA.exists(),
        f"não encontrei {ENTRADA_MATRIZ_BUSBRA}; rode features.py do BUS-BRA antes "
        "(o Protocolo C depende de os dois vetores terem exatamente o mesmo D)",
    )
    matriz_busbra = np.load(ENTRADA_MATRIZ_BUSBRA)
    exigir(
        matriz.shape[1] == matriz_busbra.shape[1],
        f"dimensão do vetor de características do BrEaST ({matriz.shape[1]}) não bate "
        f"com a do BUS-BRA ({matriz_busbra.shape[1]}) — features.py e features_breast.py "
        "divergiram; o Protocolo C não pode aplicar o modelo treinado no BUS-BRA ao "
        "BrEaST se os vetores não tiverem a mesma dimensão",
    )


def numero(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def resumir(matriz: np.ndarray) -> None:
    print(f"\n{numero(matriz.shape[0])} imagens do BrEaST x {matriz.shape[1]} "
          "características (mesmo D do BUS-BRA, conferido).")


def main() -> None:
    tabela = ler_indice_breast()
    exigir(
        len(tabela) == LINHAS_ESPERADAS,
        f"esperava {LINHAS_ESPERADAS} linhas em indice_breast.csv, encontrei {len(tabela)}",
    )

    print(f"extraindo características de {numero(len(tabela))} imagens do BrEaST "
          f"(redimensionadas para {TAMANHO[0]}x{TAMANHO[1]}, mesma função de features.py)...")
    matriz = extrair_todas(tabela)

    verificar_sanidade_matriz(matriz, tabela)   # shape[0]==len(tabela), sem NaN/Inf (features.py)
    verificar_dimensao_igual_busbra(matriz)      # D igual ao BUS-BRA (só deste script)

    SAIDA_MATRIZ.parent.mkdir(parents=True, exist_ok=True)
    np.save(SAIDA_MATRIZ, matriz)
    tabela[COLUNAS_IDS].to_csv(SAIDA_IDS, index=False)

    print(f"\nmatriz salva em {SAIDA_MATRIZ.relative_to(RAIZ_TCC).as_posix()}")
    print(f"ids alinhados salvos em {SAIDA_IDS.relative_to(RAIZ_TCC).as_posix()}")
    resumir(matriz)


if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
