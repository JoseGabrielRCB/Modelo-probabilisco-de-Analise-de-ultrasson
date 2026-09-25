"""Caminho de comparacao: caracteristicas extraidas por uma ResNet18 pre-treinada em
ImageNet (torchvision), congelada e usada so como extrator — sem nenhum treino.

Nao substitui o caminho classico (`comum/features.py`): le o mesmo
`indice_particionado.csv`, mas escreve em arquivos proprios.

Para cada imagem:
    1. converte para RGB (o canal cinza e replicado em R=G=B);
    2. redimensiona para 224x224;
    3. escala para [0,1] e normaliza com media/desvio do ImageNet;
    4. passa pela ResNet18 em `.eval()` e `torch.no_grad()`, com a `fc` trocada por
       `nn.Identity()`: o vetor e a saida do avgpool, 512 numeros por imagem.

Escreve, alinhados linha a linha:
    - `dados_processados/caracteristicas_resnet.npy`      — matriz N x 512 (float64)
    - `dados_processados/caracteristicas_resnet_ids.csv`  — id, paciente, fold, rotulo,
      base, imagem

Uso:
    python src/resnet/features_resnet.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.models import ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../03-codigo/src
from comum.indexar import numero

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA = RAIZ_CODIGO / "dados_processados" / "indice_particionado.csv"
SAIDA_MATRIZ = RAIZ_CODIGO / "dados_processados" / "caracteristicas_resnet.npy"
SAIDA_IDS = RAIZ_CODIGO / "dados_processados" / "caracteristicas_resnet_ids.csv"

# Mesmas colunas de ids do caminho classico (comum/features.py)
COLUNAS_IDS = ["id", "paciente", "fold", "rotulo", "base", "imagem"]

TAMANHO = (224, 224)  # (largura, altura); entrada padrao da ResNet no ImageNet
MEDIA_IMAGENET = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DESVIO_IMAGENET = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TAMANHO_LOTE = 32  # lotes so por velocidade em CPU; nao afeta o resultado
INTERVALO_PROGRESSO = 200


def carregar_modelo() -> torch.nn.Module:
    """ResNet18 com pesos IMAGENET1K_V1, sem a `fc`, em `.eval()` e com pesos congelados."""
    modelo = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    modelo.fc = torch.nn.Identity()
    modelo.eval()
    for parametro in modelo.parameters():
        parametro.requires_grad_(False)
    return modelo


def preprocessar_imagem(caminho: Path) -> torch.Tensor:
    """Imagem -> tensor (3, 224, 224) em RGB, redimensionado e normalizado."""
    with Image.open(caminho) as img:
        rgb = np.array(img.convert("RGB").resize(TAMANHO, Image.BILINEAR))
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return (tensor - MEDIA_IMAGENET) / DESVIO_IMAGENET


def extrair_todas(tabela: pd.DataFrame, modelo: torch.nn.Module) -> np.ndarray:
    """Vetor de 512 caracteristicas por linha, em lotes, na ordem da tabela."""
    caminhos = [RAIZ_TCC / linha.imagem for linha in tabela.itertuples(index=False)]
    total = len(caminhos)
    vetores = []
    ultimo_impresso = 0
    inicio = time.time()

    with torch.no_grad():
        for inicio_lote in range(0, total, TAMANHO_LOTE):
            lote_caminhos = caminhos[inicio_lote : inicio_lote + TAMANHO_LOTE]
            entrada = torch.stack([preprocessar_imagem(c) for c in lote_caminhos], dim=0)
            vetores.append(modelo(entrada).numpy())  # (tamanho_do_lote, 512)

            processadas = inicio_lote + len(lote_caminhos)
            if processadas - ultimo_impresso >= INTERVALO_PROGRESSO or processadas == total:
                print(f"  {processadas}/{total} imagens processadas ({time.time() - inicio:.0f}s)")
                ultimo_impresso = processadas

    return np.vstack(vetores).astype(np.float64)


def main() -> None:
    tabela = pd.read_csv(ENTRADA, dtype={"birads": str})

    print("carregando ResNet18 (ImageNet)")
    modelo = carregar_modelo()

    print(
        f"extraindo carct: {numero(len(tabela))} imagens "
        f"({TAMANHO[0]}x{TAMANHO[1]}, lotes de {TAMANHO_LOTE})"
    )
    matriz = extrair_todas(tabela, modelo)

    SAIDA_MATRIZ.parent.mkdir(parents=True, exist_ok=True)
    np.save(SAIDA_MATRIZ, matriz)
    tabela[COLUNAS_IDS].to_csv(SAIDA_IDS, index=False)

    print(f"\nmatriz e ids salvos em {SAIDA_MATRIZ.parent.relative_to(RAIZ_TCC).as_posix()}/")
    print(f"\n{numero(matriz.shape[0])} imagens x {matriz.shape[1]} carct (avgpool da ResNet18)")


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
