"""Etapa 4 do pipeline: calcula um vetor de caracteristicas classicas de imagem para cada
linha de `indice_particionado.csv`.

Le so `indice_particionado.csv` (nunca os dados brutos diretamente — mesma regra de
fronteira das etapas anteriores) e, para cada imagem, calcula um vetor de caracteristicas
a partir da imagem em tons de cinza inteira (sem recortar pela mascara).

Decisao consciente do projeto, registrada aqui de proposito: usar a imagem inteira, nao um
recorte pela mascara da lesao. Isso mistura caracteristicas do fundo/tecido ao redor da
lesao junto com as da lesao em si, o que e uma limitacao discutida no TCC — nao um bug a
corrigir. Um recorte pela mascara e um experimento futuro possivel, nao o baseline atual.

Categorias de caracteristicas (contagem exata impressa no resumo final, ver `resumir`):
    - estatistica de intensidade (11): media, desvio padrao, minimo, maximo, mediana,
      assimetria, curtose, percentis 10/25/75/90;
    - textura GLCM/Haralick (6): contraste, dissimilaridade, homogeneidade, energia,
      correlacao, ASM — cada uma ja e a media entre os 4 angulos (0°, 45°, 90°, 135°),
      distancia 1 pixel;
    - LBP uniforme, P=8, R=1 (10): histograma normalizado (10 bins possiveis do padrao
      uniforme com P=8);
    - bordas via Sobel (3): media, desvio padrao e maximo da magnitude do gradiente;
    - grade 4x4 (32): media e desvio padrao de intensidade de cada um dos 16 blocos
      iguais em que a imagem e dividida.

Todas as imagens sao redimensionadas para um tamanho fixo (`TAMANHO`, 256x256) antes do
calculo, para que todo vetor tenha a mesma dimensao independente do tamanho original da
imagem no disco (o BUS-BRA tem imagens de tamanhos variados). Essa e uma escolha de
padronizacao, documentada aqui: 256x256 e grande o bastante para preservar textura visivel
a olho e pequeno o bastante para manter o calculo rapido em CPU.

Escreve dois arquivos, alinhados linha a linha na mesma ordem:
    - `dados_processados/caracteristicas.npy`  — matriz N x D (float64)
    - `dados_processados/caracteristicas_ids.csv` — colunas id, paciente, fold, rotulo
      (e mais algumas, ver `COLUNAS_IDS`), para mapear cada linha da matriz de volta ao
      paciente/rotulo/fold sem ambiguidade. Nenhum script depois deste (`experimento.py`)
      deve precisar reabrir `indice_particionado.csv` para saber a quem uma linha da
      matriz pertence.

Uso:
    python src/comum/features.py
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

# Caminhos: derivados da posicao deste arquivo (ver indexar.py)

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA = RAIZ_CODIGO / "dados_processados" / "indice_particionado.csv"
SAIDA_MATRIZ = RAIZ_CODIGO / "dados_processados" / "caracteristicas.npy"
SAIDA_IDS = RAIZ_CODIGO / "dados_processados" / "caracteristicas_ids.csv"

# Colunas que ligam cada linha da matriz a paciente/rotulo/fold
COLUNAS_IDS = ["id", "paciente", "fold", "rotulo", "base", "imagem"]

# Parametros de extracao

TAMANHO = (256, 256)  # (largura, altura); ver docstring do modulo

# GLCM: distancia 1 pixel, 4 angulos (0, 45, 90, 135 graus); usa a media entre os angulos
GLCM_DISTANCIAS = [1]
GLCM_ANGULOS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_NIVEIS = 256  # imagem ja esta em uint8 (0-255)
GLCM_PROPRIEDADES = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
]

# LBP uniforme: P=8, R=1 -> P+2 = 10 padroes (P+1 uniformes + 1 nao uniforme)
LBP_P = 8
LBP_R = 1
LBP_METODO = "uniform"
LBP_N_BINS = LBP_P + 2

GRADE_N = 4  # grade 4x4 = 16 blocos


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com mensagem clara se a condicao nao valer."""
    if not condicao:
        raise ValueError(mensagem)


def ler_indice_particionado() -> pd.DataFrame:
    exigir(
        ENTRADA.exists(),
        f"não encontrei {ENTRADA}; rode indexar.py, particionar.py e verificar.py antes",
    )
    return pd.read_csv(ENTRADA, dtype={"birads": str})


def carregar_imagem_cinza(caminho: Path) -> tuple[np.ndarray, np.ndarray]:
    """Abre a imagem em cinza no TAMANHO fixo; devolve uint8 (GLCM/LBP) e float64 (resto)."""
    with Image.open(caminho) as img:
        cinza = img.convert("L").resize(TAMANHO, Image.BILINEAR)
    # np.array forca copia: buffer do Pillow pode ser so leitura e o graycomatrix exige gravavel
    array_uint8 = np.array(cinza, dtype=np.uint8)
    array_float = array_uint8.astype(np.float64) / 255.0
    return array_uint8, array_float


def features_intensidade(imagem_float: np.ndarray) -> np.ndarray:
    """11 caracteristicas: media, desvio, min, max, mediana, assimetria, curtose, percentis."""
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
    """6 caracteristicas de textura GLCM/Haralick, cada uma com media entre os 4 angulos."""
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
        # graycoprops devolve (n_distancias, n_angulos); com 1 distancia, tira a media dos angulos
        valores.append(graycoprops(matriz, propriedade)[0].mean())
    return np.array(valores)


def features_lbp(imagem_uint8: np.ndarray) -> np.ndarray:
    """10 caracteristicas: histograma normalizado do LBP uniforme (P=8, R=1)."""
    padroes = local_binary_pattern(imagem_uint8, P=LBP_P, R=LBP_R, method=LBP_METODO)
    histograma, _ = np.histogram(
        padroes, bins=LBP_N_BINS, range=(0, LBP_N_BINS), density=False
    )
    total = histograma.sum()
    exigir(total > 0, "histograma LBP vazio (imagem sem pixels? não deveria acontecer)")
    return histograma.astype(np.float64) / total


def features_sobel(imagem_float: np.ndarray) -> np.ndarray:
    """3 caracteristicas: media, desvio padrao e maximo da magnitude do gradiente (Sobel)."""
    magnitude = sobel(imagem_float)
    return np.array([magnitude.mean(), magnitude.std(), magnitude.max()])


def features_grade(imagem_float: np.ndarray) -> np.ndarray:
    """32 caracteristicas: media e desvio padrao de cada um dos 16 blocos da grade 4x4."""
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
    """Vetor completo de uma imagem: concatena as 5 categorias na ordem do modulo."""
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
    """Extrai o vetor de cada linha do indice, na mesma ordem; mostra progresso a cada 200."""
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
    """Confere a matriz final antes de salvar."""
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
        "estatistica de intensidade": 11,
        "GLCM/Haralick": len(GLCM_PROPRIEDADES),
        "LBP uniforme (histograma)": LBP_N_BINS,
        "bordas (Sobel)": 3,
        f"grade {GRADE_N}x{GRADE_N} (media+desvio por bloco)": 2 * GRADE_N * GRADE_N,
    }
    total_esperado = sum(contagens.values())
    exigir(
        matriz.shape[1] == total_esperado,
        f"dimensão da matriz ({matriz.shape[1]}) não bate com a soma das categorias "
        f"documentadas ({total_esperado}) — a lista de categorias ficou desatualizada",
    )
    print(f"\n{numero(matriz.shape[0])} imagens x {matriz.shape[1]} carct:")
    for nome, quantidade in contagens.items():
        print(f"  {nome}: {quantidade}")


def main() -> None:
    tabela = ler_indice_particionado()

    print(f"extraindo carct: {numero(len(tabela))} imagens ({TAMANHO[0]}x{TAMANHO[1]})")
    matriz = extrair_todas(tabela)

    verificar_sanidade(matriz, tabela)

    SAIDA_MATRIZ.parent.mkdir(parents=True, exist_ok=True)
    np.save(SAIDA_MATRIZ, matriz)
    tabela[COLUNAS_IDS].to_csv(SAIDA_IDS, index=False)

    print(f"\nmatriz e ids salvos em {SAIDA_MATRIZ.parent.relative_to(RAIZ_TCC).as_posix()}/")
    resumir(matriz)


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
