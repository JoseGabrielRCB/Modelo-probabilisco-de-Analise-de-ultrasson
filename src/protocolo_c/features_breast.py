"""Etapa 3 (BrEaST) do Protocolo C: calcula o MESMO vetor de caracteristicas do BUS-BRA.

Usa `extrair_vetor` de `comum/features.py` (mesmas 5 categorias, mesma ordem, mesmo
redimensionamento), para a comparacao BUS-BRA x BrEaST medir o mesmo tipo de
caracteristica nas duas bases. Le so `dados_processados/breast/indice_breast.csv` e escreve:
    - `dados_processados/breast/caracteristicas_breast.npy`      — matriz 252 x D
    - `dados_processados/breast/caracteristicas_breast_ids.csv`  — id, paciente, rotulo

Uso:
    python src/protocolo_c/features_breast.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from comum.features import TAMANHO, extrair_vetor
from comum.features import verificar_sanidade as verificar_sanidade_matriz
from comum.indexar import numero

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA = RAIZ_CODIGO / "dados_processados" / "breast" / "indice_breast.csv"
SAIDA_MATRIZ = RAIZ_CODIGO / "dados_processados" / "breast" / "caracteristicas_breast.npy"
SAIDA_IDS = RAIZ_CODIGO / "dados_processados" / "breast" / "caracteristicas_breast_ids.csv"

COLUNAS_IDS = ["id", "paciente", "rotulo"]


def extrair_todas(tabela: pd.DataFrame) -> np.ndarray:
    """Vetor de caracteristicas de cada linha, na ordem da tabela, via `extrair_vetor`."""
    vetores = []
    inicio = time.time()
    total = len(tabela)
    for posicao, linha in enumerate(tabela.itertuples(index=False), start=1):
        vetores.append(extrair_vetor(RAIZ_TCC / linha.imagem))
        if posicao % 50 == 0 or posicao == total:
            print(f"  {posicao}/{total} imagens processadas ({time.time() - inicio:.0f}s)")
    return np.vstack(vetores)


def main() -> None:
    tabela = pd.read_csv(ENTRADA, dtype={"birads": str})

    print(f"extraindo carct do BrEaST: {numero(len(tabela))} imagens ({TAMANHO[0]}x{TAMANHO[1]})")
    matriz = extrair_todas(tabela)
    verificar_sanidade_matriz(matriz, tabela)   # uma linha por imagem, sem NaN/Inf

    SAIDA_MATRIZ.parent.mkdir(parents=True, exist_ok=True)
    np.save(SAIDA_MATRIZ, matriz)
    tabela[COLUNAS_IDS].to_csv(SAIDA_IDS, index=False)

    print(f"\nmatriz e ids salvos em {SAIDA_MATRIZ.parent.relative_to(RAIZ_TCC).as_posix()}/")
    print(f"\nBrEaST: {numero(matriz.shape[0])} imagens x {matriz.shape[1]} carct")


if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        # Usar esse metodo é um pouco estranho , mas foi nescessario para poder validar um erro ocorrido e tambem por questao de docuemntao
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
