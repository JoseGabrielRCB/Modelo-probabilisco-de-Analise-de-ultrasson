"""Etapa 4 do pipeline: calcula um vetor de características clássicas de imagem para cada
linha de `indice_particionado.csv`.

Lê só `indice_particionado.csv` (nunca os dados brutos diretamente — mesma regra de
fronteira das etapas anteriores) e, para cada imagem, calcula um vetor de características
a partir da imagem em tons de cinza inteira (sem recortar pela máscara).

Decisão consciente do projeto, registrada aqui de propósito: usar a imagem inteira, não um
recorte pela máscara da lesão. Isso mistura características do fundo/tecido ao redor da
lesão junto com as da lesão em si, o que é uma limitação discutida no TCC — não um bug a
corrigir. Um recorte pela máscara é um experimento futuro possível, não o baseline atual.

Categorias de características (contagem exata impressa no resumo final, ver `resumir`):
    - estatística de intensidade (11): média, desvio padrão, mínimo, máximo, mediana,
      assimetria, curtose, percentis 10/25/75/90;
    - textura GLCM/Haralick (6): contraste, dissimilaridade, homogeneidade, energia,
      correlação, ASM — cada uma já é a média entre os 4 ângulos (0°, 45°, 90°, 135°),
      distância 1 pixel;
    - LBP uniforme, P=8, R=1 (10): histograma normalizado (10 bins possíveis do padrão
      uniforme com P=8);
    - bordas via Sobel (3): média, desvio padrão e máximo da magnitude do gradiente;
    - grade 4x4 (32): média e desvio padrão de intensidade de cada um dos 16 blocos
      iguais em que a imagem é dividida.

Todas as imagens são redimensionadas para um tamanho fixo (`TAMANHO`, 256x256) antes do
cálculo, para que todo vetor tenha a mesma dimensão independente do tamanho original da
imagem no disco (o BUS-BRA tem imagens de tamanhos variados). Essa é uma escolha de
padronização, documentada aqui: 256x256 é grande o bastante para preservar textura visível
a olho e pequeno o bastante para manter o cálculo rápido em CPU.

Escreve dois arquivos, alinhados linha a linha na mesma ordem:
    - `dados_processados/caracteristicas.npy`  — matriz N x D (float64)
    - `dados_processados/caracteristicas_ids.csv` — colunas id, paciente, fold, rotulo
      (e mais algumas, ver `COLUNAS_IDS`), para mapear cada linha da matriz de volta ao
      paciente/rótulo/fold sem ambiguidade. Nenhum script depois deste (`experimento.py`)
      deve precisar reabrir `indice_particionado.csv` para saber a quem uma linha da
      matriz pertence.

Uso:
    python src/features.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as sp_stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel

# ---------------------------------------------------------------------------
# Caminhos — tudo derivado da posição deste arquivo (ver indexar.py).
# ---------------------------------------------------------------------------

RAIZ_CODIGO = Path(__file__).resolve().parents[1]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA = RAIZ_CODIGO / "dados_processados" / "indice_particionado.csv"
SAIDA_MATRIZ = RAIZ_CODIGO / "dados_processados" / "caracteristicas.npy"
SAIDA_IDS = RAIZ_CODIGO / "dados_processados" / "caracteristicas_ids.csv"

# Colunas do arquivo de acompanhamento — o suficiente para mapear qualquer linha da
# matriz de volta ao paciente/rótulo/fold sem precisar reabrir indice_particionado.csv.
COLUNAS_IDS = ["id", "paciente", "fold", "rotulo", "base", "imagem"]

# ---------------------------------------------------------------------------
# Parâmetros de extração — documentados aqui, não escondidos dentro das funções.
# ---------------------------------------------------------------------------

TAMANHO = (256, 256)  # (largura, altura); ver justificativa no docstring do módulo

# GLCM: distância 1 pixel, 4 ângulos (0°, 45°, 90°, 135°); as propriedades saem como a
# média entre os 4 ângulos, não um vetor separado por ângulo — reduz dimensão mantendo
# a informação de textura "média" da imagem.
GLCM_DISTANCIAS = [1]
GLCM_ANGULOS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_NIVEIS = 256  # imagem já está em uint8 (0-255)
GLCM_PROPRIEDADES = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
]

# LBP uniforme: P=8, R=1 -> P+2 = 10 padrões possíveis (P+1 uniformes + 1 "não-uniforme").
LBP_P = 8
LBP_R = 1
LBP_METODO = "uniform"
LBP_N_BINS = LBP_P + 2

GRADE_N = 4  # grade 4x4 = 16 blocos


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com uma mensagem clara se a condição não valer. Mesma função das
    etapas anteriores: parar com erro, nunca seguir em frente com dado suspeito."""
    if not condicao:
        raise ValueError(mensagem)


def ler_indice_particionado() -> pd.DataFrame:
    exigir(
        ENTRADA.exists(),
        f"não encontrei {ENTRADA}; rode indexar.py, particionar.py e verificar.py antes",
    )
    return pd.read_csv(ENTRADA, dtype={"birads": str})


def carregar_imagem_cinza(caminho: Path) -> tuple[np.ndarray, np.ndarray]:
    """Abre a imagem, converte para tons de cinza e redimensiona para TAMANHO.

    Devolve dois arrays da mesma imagem redimensionada: um em uint8 (0-255, para GLCM e
    LBP) e outro em float64 (0-1, para as estatísticas e o Sobel).
    """
    with Image.open(caminho) as img:
        cinza = img.convert("L").resize(TAMANHO, Image.BILINEAR)
    # np.array (nao np.asarray) forca uma copia: o buffer que o Pillow devolve pode vir
    # somente-leitura, e o graycomatrix (Cython) exige um array gravavel.
    array_uint8 = np.array(cinza, dtype=np.uint8)
    array_float = array_uint8.astype(np.float64) / 255.0
    return array_uint8, array_float


def features_intensidade(imagem_float: np.ndarray) -> np.ndarray:
    """11 características: média, desvio padrão, mínimo, máximo, mediana, assimetria,
    curtose, percentis 10/25/75/90."""
    achatada = imagem_float.ravel()
    percentis = np.percentile(achatada, [10, 25, 75, 90])
    return np.array(
        [
            achatada.mean(),
            achatada.std(),
            achatada.min(),
            achatada.max(),
            np.median(achatada),
            sp_stats.skew(achatada),
            sp_stats.kurtosis(achatada),
            *percentis,
        ]
    )


def features_glcm(imagem_uint8: np.ndarray) -> np.ndarray:
    """6 características de textura (GLCM/Haralick), cada uma já reduzida à média entre
    os 4 ângulos calculados."""
    matriz = graycomatrix(
        imagem_uint8,
        distances=GLCM_DISTANCIAS,
        angles=GLCM_ANGULOS,
        levels=GLCM_NIVEIS,
        symmetric=True,
        normed=True,
    )
    valores = []
    for propriedade in GLCM_PROPRIEDADES:
        # graycoprops devolve forma (n_distancias, n_angulos); com 1 distância, tiramos
        # a média entre os ângulos.
        valores.append(graycoprops(matriz, propriedade)[0].mean())
    return np.array(valores)


def features_lbp(imagem_uint8: np.ndarray) -> np.ndarray:
    """10 características: histograma normalizado do LBP uniforme (P=8, R=1)."""
    padroes = local_binary_pattern(imagem_uint8, P=LBP_P, R=LBP_R, method=LBP_METODO)
    histograma, _ = np.histogram(
        padroes, bins=LBP_N_BINS, range=(0, LBP_N_BINS), density=False
    )
    total = histograma.sum()
    exigir(total > 0, "histograma LBP vazio (imagem sem pixels? não deveria acontecer)")
    return histograma.astype(np.float64) / total


def features_sobel(imagem_float: np.ndarray) -> np.ndarray:
    """3 características: média, desvio padrão e máximo da magnitude do gradiente
    (filtro de Sobel)."""
    magnitude = sobel(imagem_float)
    return np.array([magnitude.mean(), magnitude.std(), magnitude.max()])


def features_grade(imagem_float: np.ndarray) -> np.ndarray:
    """32 características: média e desvio padrão de intensidade de cada um dos 16 blocos
    de uma grade 4x4 (a imagem já está em TAMANHO fixo, então os blocos têm sempre o
    mesmo tamanho em pixels)."""
    altura, largura = imagem_float.shape
    exigir(
        altura % GRADE_N == 0 and largura % GRADE_N == 0,
        f"TAMANHO {TAMANHO} não é divisível por {GRADE_N}x{GRADE_N} sem resto",
    )
    passo_altura = altura // GRADE_N
    passo_largura = largura // GRADE_N

    valores = []
    for linha in range(GRADE_N):
        for coluna in range(GRADE_N):
            bloco = imagem_float[
                linha * passo_altura : (linha + 1) * passo_altura,
                coluna * passo_largura : (coluna + 1) * passo_largura,
            ]
            valores.append(bloco.mean())
            valores.append(bloco.std())
    return np.array(valores)


def extrair_vetor(caminho_imagem: Path) -> np.ndarray:
    """Monta o vetor de características completo de uma imagem, concatenando as 5
    categorias na ordem documentada no módulo."""
    imagem_uint8, imagem_float = carregar_imagem_cinza(caminho_imagem)
    return np.concatenate(
        [
            features_intensidade(imagem_float),
            features_glcm(imagem_uint8),
            features_lbp(imagem_uint8),
            features_sobel(imagem_float),
            features_grade(imagem_float),
        ]
    )


def extrair_todas(tabela: pd.DataFrame) -> np.ndarray:
    """Extrai o vetor de características de cada linha do índice, na mesma ordem das
    linhas da tabela. Imprime progresso a cada 200 imagens — a extração para as ~1.875
    imagens do BUS-BRA leva alguns minutos em CPU, e um script mudo por minutos parece
    travado."""
    vetores = []
    inicio = time.time()
    total = len(tabela)
    for posicao, linha in enumerate(tabela.itertuples(index=False), start=1):
        caminho = RAIZ_TCC / linha.imagem
        exigir(caminho.exists(), f"imagem não encontrada no disco: {caminho}")
        vetores.append(extrair_vetor(caminho))
        if posicao % 200 == 0 or posicao == total:
            decorrido = time.time() - inicio
            print(f"  {posicao}/{total} imagens processadas ({decorrido:.0f}s)")
    return np.vstack(vetores)


def verificar_sanidade(matriz: np.ndarray, tabela: pd.DataFrame) -> None:
    """Confere a matriz final antes de salvar qualquer coisa."""
    exigir(
        matriz.shape[0] == len(tabela),
        f"matriz tem {matriz.shape[0]} linhas, esperava {len(tabela)} (uma por imagem)",
    )
    exigir(
        not np.isnan(matriz).any(),
        f"{int(np.isnan(matriz).sum())} valor(es) NaN na matriz de características",
    )
    exigir(
        not np.isinf(matriz).any(),
        f"{int(np.isinf(matriz).sum())} valor(es) Inf na matriz de características",
    )


def numero(valor: int) -> str:
    """Formata um inteiro com ponto de milhar (1875 -> '1.875')."""
    return f"{valor:,}".replace(",", ".")


def resumir(matriz: np.ndarray) -> None:
    contagens = {
        "estatística de intensidade": 11,
        "GLCM/Haralick": len(GLCM_PROPRIEDADES),
        "LBP uniforme (histograma)": LBP_N_BINS,
        "bordas (Sobel)": 3,
        f"grade {GRADE_N}x{GRADE_N} (média+desvio por bloco)": 2 * GRADE_N * GRADE_N,
    }
    total_esperado = sum(contagens.values())
    exigir(
        matriz.shape[1] == total_esperado,
        f"dimensão da matriz ({matriz.shape[1]}) não bate com a soma das categorias "
        f"documentadas ({total_esperado}) — a lista de categorias ficou desatualizada",
    )
    print(f"\n{numero(matriz.shape[0])} imagens x {matriz.shape[1]} características:")
    for nome, quantidade in contagens.items():
        print(f"  {nome}: {quantidade}")


def main() -> None:
    tabela = ler_indice_particionado()

    print(f"extraindo características de {numero(len(tabela))} imagens "
          f"(redimensionadas para {TAMANHO[0]}x{TAMANHO[1]})...")
    matriz = extrair_todas(tabela)

    verificar_sanidade(matriz, tabela)

    SAIDA_MATRIZ.parent.mkdir(parents=True, exist_ok=True)
    np.save(SAIDA_MATRIZ, matriz)
    tabela[COLUNAS_IDS].to_csv(SAIDA_IDS, index=False)

    print(f"\nmatriz salva em {SAIDA_MATRIZ.relative_to(RAIZ_TCC).as_posix()}")
    print(f"ids alinhados salvos em {SAIDA_IDS.relative_to(RAIZ_TCC).as_posix()}")
    resumir(matriz)


if __name__ == "__main__":
    # O console do Windows costuma abrir em cp1252 e comeria os acentos das mensagens.
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
