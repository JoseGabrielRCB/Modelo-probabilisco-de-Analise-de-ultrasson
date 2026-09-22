"""Caminho alternativo de extração de características: ResNet18 pré-treinada em
ImageNet (via torchvision), congelada, usada só como extrator -- sem nenhum treino.

Existe para COMPARAR com o caminho clássico (`features.py`, características de
scikit-image + regressão logística) -- não substitui nem sobrescreve nada do que ele
produz. Lê a mesma fonte (`dados_processados/indice_particionado.csv`), mesma regra de
fronteira das outras etapas (nunca lê os dados brutos diretamente), mas escreve em
arquivos com nomes diferentes (`caracteristicas_resnet.npy` /
`caracteristicas_resnet_ids.csv`), então rodar este script nunca risca o resultado
clássico já verificado (AUROC 0,777 [0,744; 0,805] por paciente, ver
`resultados/metricas.md`).

Para cada imagem:
    1. Abre com PIL, converte para RGB (a imagem de origem é escala de cinza -- o
       `.convert("RGB")` do Pillow já replica o canal L nos 3 canais R=G=B sozinho,
       conferido manualmente antes de assumir isso aqui).
    2. Redimensiona para 224x224 (tamanho de entrada padrão da ResNet no ImageNet).
    3. Normaliza com média/desvio padrão do ImageNet
       ([0,485;0,456;0,406] / [0,229;0,224;0,225]), depois de escalar para [0,1].
    4. Passa pela ResNet18 pré-treinada, em modo `.eval()`, dentro de `torch.no_grad()`,
       com a camada `fc` trocada por `nn.Identity()` -- ou seja, o vetor salvo é a saída
       do `avgpool` (penúltima camada), 512 números por imagem. Rede congelada o tempo
       todo: nenhum parâmetro é atualizado, nenhum gradiente é calculado.

Escreve dois arquivos, alinhados linha a linha na mesma ordem (mesmo formato de colunas
de `caracteristicas_ids.csv` do caminho clássico, para os scripts das etapas seguintes
tratarem os dois caminhos de forma simétrica):
    - `dados_processados/caracteristicas_resnet.npy`      -- matriz N x 512 (float64)
    - `dados_processados/caracteristicas_resnet_ids.csv`  -- colunas id, paciente, fold,
      rotulo, base, imagem

## Ponte numpy <-> torch: por que não uso `ToTensor()`/`tensor.numpy()`

A versão de PyTorch usada aqui (2.0.1, CPU) foi compilada contra a ABI do NumPy 1.x. O
resto do pipeline (features.py, sklearn) já usa NumPy 2.x instalado no ambiente -- e
trocar a versão do NumPy só para o PyTorch quebraria o caminho clássico, o que é
justamente o que este script não pode fazer. `torchvision.transforms.ToTensor()` e
`tensor.numpy()` dependem dessa ponte C (`_ARRAY_API`) e falham em runtime nessa
combinação de versões (`RuntimeError: Numpy is not available`), mesmo com o import
funcionando sem erro.

Solução: contornar a ponte numpy<->torch inteiramente, nos dois sentidos --
    - imagem -> tensor: `Image.tobytes()` (bytes puros do Pillow) -> `torch.frombuffer`
      (lê o buffer diretamente, sem passar por um ndarray do NumPy);
    - tensor -> matriz final: `.tolist()` (lista Python pura) -> `np.array(...)`, em vez
      de `.numpy()`.
Isso não é só um contorno específico deste ambiente sandboxed: qualquer combinação de
PyTorch pré-2.x-numpy-2.0 com NumPy 2.x instalado (comum em setups recentes) bate no
mesmo problema, então o contorno é a escolha certa mesmo fora deste ambiente.

## Pesos pré-treinados: por que um espelho local

O caminho padrão (`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`)
baixa o arquivo de `download.pytorch.org` na primeira vez e guarda em cache local
(`~/.cache/torch/hub/checkpoints/`) -- é o que `carregar_modelo()` tenta primeiro, e é
tudo que precisa rodar em qualquer máquina com internet normal (inclusive a máquina do
autor deste TCC, fora deste ambiente de desenvolvimento).

Neste ambiente de desenvolvimento (sandboxed), porém, `download.pytorch.org` é bloqueado
por política de rede (confirmado: toda tentativa de CONNECT a esse host retorna 403 do
gateway de saída, tanto para o pip install do PyTorch quanto para este download). Como
fallback, o script usa um espelho dos MESMOS pesos, hospedado como blob comum (não
LFS) em um repositório público bem estabelecido:

    URL:      https://raw.githubusercontent.com/fregu856/deeplabv3/master/pretrained_models/resnet/resnet18-5c106cde.pth
    SHA256:   5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8
    Tamanho:  46.827.520 bytes (44,66 MiB)

Esse é o checkpoint "legado" da ResNet18 da própria torchvision (nome de arquivo de
antes da API de múltiplos pesos por enum) -- mesmos pesos, mesmas métricas publicadas
pela torchvision para `ResNet18_Weights.IMAGENET1K_V1` (acc@1 = 69,758%,
acc@5 = 89,078%; ver `torchvision/models/resnet.py` no repositório oficial). O hash
SHA256 do arquivo bate exatamente com o hash embutido no próprio nome do arquivo
("5c106cde..."), convenção que o PyTorch usa há anos para permitir essa checagem de
integridade -- conferido manualmente antes de usar esse espelho neste script (ver
`_carregar_modelo_do_espelho_local`, que refaz essa conferência em toda execução, não
confia no arquivo só por estar no caminho certo).

Se nem o download normal nem o espelho local (baixado manualmente uma vez para
`~/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth` neste ambiente) estiverem
disponíveis, o script trava com uma mensagem explicando as duas opções.

Uso:
    python src/resnet/features_resnet.py
"""

from __future__ import annotations

import hashlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.models import ResNet18_Weights

# ---------------------------------------------------------------------------
# Caminhos -- tudo derivado da posição deste arquivo (ver indexar.py).
# ---------------------------------------------------------------------------

RAIZ_CODIGO = Path(__file__).resolve().parents[2]   # .../TCC-Ultrassom/03-codigo
RAIZ_TCC = RAIZ_CODIGO.parent                       # .../TCC-Ultrassom

ENTRADA = RAIZ_CODIGO / "dados_processados" / "indice_particionado.csv"
SAIDA_MATRIZ = RAIZ_CODIGO / "dados_processados" / "caracteristicas_resnet.npy"
SAIDA_IDS = RAIZ_CODIGO / "dados_processados" / "caracteristicas_resnet_ids.csv"

# Mesmas colunas de acompanhamento do caminho clássico (features.py) -- mantém os dois
# caminhos simétricos para quem for ler os dois arquivos de ids.
COLUNAS_IDS = ["id", "paciente", "fold", "rotulo", "base", "imagem"]

# ---------------------------------------------------------------------------
# Parâmetros de extração -- documentados aqui, não escondidos dentro das funções.
# ---------------------------------------------------------------------------

TAMANHO = (224, 224)  # (largura, altura); entrada padrão da ResNet no ImageNet
MEDIA_IMAGENET = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DESVIO_IMAGENET = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
N_CARACTERISTICAS = 512  # saída do avgpool da ResNet18 (antes da fc removida)

TAMANHO_LOTE = 32  # processa em lotes só por velocidade em CPU; não afeta o resultado
INTERVALO_PROGRESSO = 200  # imprime progresso a cada ~200 imagens, ver docstring

# --- Espelho local dos pesos, ver docstring do módulo, seção "Pesos pré-treinados" ---
NOME_ARQUIVO_ESPELHO = "resnet18-5c106cde.pth"
URL_ESPELHO = (
    "https://raw.githubusercontent.com/fregu856/deeplabv3/master/"
    "pretrained_models/resnet/resnet18-5c106cde.pth"
)
SHA256_ESPERADO_ESPELHO = "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8"


def exigir(condicao: bool, mensagem: str) -> None:
    """Trava o script com uma mensagem clara se a condição não valer. Mesma função das
    outras etapas do pipeline: parar com erro, nunca seguir em frente com dado suspeito."""
    if not condicao:
        raise ValueError(mensagem)


def ler_indice_particionado() -> pd.DataFrame:
    exigir(
        ENTRADA.exists(),
        f"não encontrei {ENTRADA}; rode indexar.py, particionar.py e verificar.py antes",
    )
    return pd.read_csv(ENTRADA, dtype={"birads": str})


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def _carregar_modelo_do_espelho_local(erro_download: Exception) -> torch.nn.Module:
    """Fallback: carrega os pesos do espelho local, conferindo o hash SHA256 antes de
    confiar no arquivo -- ver docstring do módulo, seção "Pesos pré-treinados"."""
    caminho_espelho = Path(torch.hub.get_dir()) / "checkpoints" / NOME_ARQUIVO_ESPELHO
    exigir(
        caminho_espelho.exists(),
        "não consegui baixar os pesos da ResNet18 de download.pytorch.org "
        f"(erro: {erro_download!r}) e não encontrei o espelho local em "
        f"{caminho_espelho}. Duas opções: (1) rodar este script numa máquina com "
        f"acesso normal à internet -- a torchvision baixa sozinha; ou (2) baixar "
        f"manualmente {URL_ESPELHO} e salvar nesse caminho (conferindo o SHA256 "
        f"{SHA256_ESPERADO_ESPELHO}). Ver docstring deste módulo para detalhes.",
    )

    sha256 = hashlib.sha256(caminho_espelho.read_bytes()).hexdigest()
    exigir(
        sha256 == SHA256_ESPERADO_ESPELHO,
        f"hash do espelho local não bate: esperava {SHA256_ESPERADO_ESPELHO}, "
        f"encontrei {sha256} -- arquivo corrompido ou não é o checkpoint esperado, "
        "não uso sem essa conferência",
    )

    modelo = torchvision.models.resnet18(weights=None)
    estado = torch.load(caminho_espelho, map_location="cpu")
    modelo.load_state_dict(estado, strict=True)
    print(
        f"AVISO: download.pytorch.org bloqueado neste ambiente ({erro_download!r}); "
        f"usando espelho local com hash conferido ({caminho_espelho}) -- mesmos pesos "
        "de ResNet18_Weights.IMAGENET1K_V1, ver docstring do módulo."
    )
    return modelo


def carregar_modelo() -> torch.nn.Module:
    """Carrega a ResNet18 pré-treinada em ImageNet, com a camada `fc` trocada por
    `nn.Identity()` (vetor de 512 números por imagem, saída do avgpool), em modo
    `.eval()` e com todos os parâmetros congelados (nenhum treino, nenhum gradiente)."""
    try:
        modelo = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        print("pesos da ResNet18 obtidos via torchvision (download.pytorch.org).")
    except Exception as erro_download:  # noqa: BLE001 -- qualquer falha de rede/URL cai no fallback
        modelo = _carregar_modelo_do_espelho_local(erro_download)

    modelo.fc = torch.nn.Identity()
    modelo.eval()
    for parametro in modelo.parameters():
        parametro.requires_grad_(False)
    return modelo


# ---------------------------------------------------------------------------
# Pré-processamento e extração
# ---------------------------------------------------------------------------

def preprocessar_imagem(caminho: Path) -> torch.Tensor:
    """Abre, converte para RGB, redimensiona e normaliza uma imagem. Devolve um tensor
    (3, 224, 224) já pronto para entrar no lote.

    Usa `Image.tobytes()` + `torch.frombuffer()` em vez de `ToTensor()` -- contorna a
    ponte numpy<->torch quebrada nesta combinação de versões, ver docstring do módulo."""
    with Image.open(caminho) as img:
        rgb = img.convert("RGB").resize(TAMANHO, Image.BILINEAR)
        bruto = rgb.tobytes()

    largura, altura = TAMANHO
    tensor = torch.frombuffer(bytearray(bruto), dtype=torch.uint8)
    tensor = tensor.reshape(altura, largura, 3).permute(2, 0, 1).contiguous().float() / 255.0
    tensor = (tensor - MEDIA_IMAGENET) / DESVIO_IMAGENET
    return tensor


def extrair_todas(tabela: pd.DataFrame, modelo: torch.nn.Module) -> np.ndarray:
    """Extrai o vetor de características de cada linha do índice, em lotes de
    TAMANHO_LOTE, na mesma ordem das linhas da tabela. Imprime progresso a cada ~200
    imagens (ver INTERVALO_PROGRESSO)."""
    caminhos = [RAIZ_TCC / linha.imagem for linha in tabela.itertuples(index=False)]
    for caminho in caminhos:
        exigir(caminho.exists(), f"imagem não encontrada no disco: {caminho}")

    total = len(caminhos)
    vetores: list[list[float]] = []
    ultimo_impresso = 0
    inicio = time.time()

    with torch.no_grad():
        for inicio_lote in range(0, total, TAMANHO_LOTE):
            lote_caminhos = caminhos[inicio_lote : inicio_lote + TAMANHO_LOTE]
            entrada = torch.stack([preprocessar_imagem(c) for c in lote_caminhos], dim=0)
            saida = modelo(entrada)  # (tamanho_do_lote, 512)
            vetores.extend(saida.tolist())  # tolist() em vez de .numpy(), ver docstring

            processadas = inicio_lote + len(lote_caminhos)
            if processadas - ultimo_impresso >= INTERVALO_PROGRESSO or processadas == total:
                decorrido = time.time() - inicio
                print(f"  {processadas}/{total} imagens processadas ({decorrido:.0f}s)")
                ultimo_impresso = processadas

    return np.array(vetores, dtype=np.float64)


def verificar_sanidade(matriz: np.ndarray, tabela: pd.DataFrame) -> None:
    """Confere a matriz final antes de salvar qualquer coisa -- mesmo critério do
    caminho clássico (features.py)."""
    exigir(
        matriz.shape == (len(tabela), N_CARACTERISTICAS),
        f"matriz tem forma {matriz.shape}, esperava ({len(tabela)}, {N_CARACTERISTICAS})",
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
    print(
        f"\n{numero(matriz.shape[0])} imagens x {matriz.shape[1]} características "
        "(saída do avgpool da ResNet18, penúltima camada, sem a fc)."
    )


def main() -> None:
    tabela = ler_indice_particionado()

    print("carregando ResNet18 pré-treinada em ImageNet (torchvision)...")
    modelo = carregar_modelo()

    print(
        f"extraindo características de {numero(len(tabela))} imagens "
        f"(redimensionadas para {TAMANHO[0]}x{TAMANHO[1]}, lotes de {TAMANHO_LOTE})..."
    )
    matriz = extrair_todas(tabela, modelo)

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

    # Aviso de incompatibilidade NumPy/PyTorch (ver docstring, seção sobre a ponte
    # numpy<->torch) -- não afeta este script, que contorna a ponte inteiramente, mas o
    # torchvision emite o aviso já no import; silenciado para não confundir quem rodar.
    warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy.*")

    try:
        main()
    except ValueError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
