# Makefile do pipeline (03-codigo/). Sintaxe GNU Make padrao -- funciona tanto com o
# `make` do Linux/macOS quanto com um `make` GNU instalado no Windows (ex.: via Git for
# Windows/MinGW ou Chocolatey), rodado dentro desta pasta (03-codigo/).
#
# Cada alvo roda `python src/<trilha>/<script>.py` (trilhas: comum/, protocolo_ab/,
# protocolo_c/, resnet/) -- sempre a partir da raiz de 03-codigo/, nunca
# de dentro de src/, porque todos os scripts resolvem seus proprios caminhos a partir da
# posicao do arquivo .py (ver o comentario "Caminhos" no topo de cada script), nao do
# diretorio de onde o comando foi chamado; ainda assim, rodar sempre daqui evita
# surpresa.
#
# Use `make PYTHON=python3 all` se o interpretador certo nao se chamar `python` no seu
# ambiente.

PYTHON ?= python

DADOS := dados_processados
RESULTADOS := resultados

.PHONY: all indexar particionar verificar features experimento metricas relatorio clean \
        all_resnet features_resnet experimento_resnet metricas_resnet

# --- Etapa 1 -----------------------------------------------------------------
indexar: $(DADOS)/indice.csv

$(DADOS)/indice.csv: src/comum/indexar.py
	$(PYTHON) src/comum/indexar.py

# --- Etapa 2 -----------------------------------------------------------------
particionar: $(DADOS)/indice_particionado.csv

$(DADOS)/indice_particionado.csv: $(DADOS)/indice.csv src/comum/particionar.py
	$(PYTHON) src/comum/particionar.py

# --- Etapa 3 -------------------------------------------------------------------
# `verificar` e o portao de auditoria: nao produz um arquivo de saida proprio (so
# confere o que ja existe), entao e sempre re-executado quando pedido -- nunca pulado
# por causa de timestamp de arquivo, ao contrario dos outros alvos.
verificar: $(DADOS)/indice_particionado.csv
	$(PYTHON) src/comum/verificar.py

# --- Etapa 4 -----------------------------------------------------------------
features: $(DADOS)/caracteristicas.npy

$(DADOS)/caracteristicas.npy: $(DADOS)/indice_particionado.csv src/comum/features.py
	$(PYTHON) src/comum/features.py

# --- Etapa 5 -----------------------------------------------------------------
experimento: $(RESULTADOS)/predicoes.csv

$(RESULTADOS)/predicoes.csv: $(DADOS)/caracteristicas.npy src/protocolo_ab/experimento.py
	$(PYTHON) src/protocolo_ab/experimento.py

# --- Etapa 6 -----------------------------------------------------------------
metricas: $(RESULTADOS)/metricas.md

$(RESULTADOS)/metricas.md: $(RESULTADOS)/predicoes.csv src/protocolo_ab/metricas.py
	$(PYTHON) src/protocolo_ab/metricas.py

# --- Etapa 7 -----------------------------------------------------------------
relatorio: $(RESULTADOS)/curva_roc.png

$(RESULTADOS)/curva_roc.png: $(RESULTADOS)/metricas.md src/protocolo_ab/relatorio.py
	$(PYTHON) src/protocolo_ab/relatorio.py

# --- Pipeline completo, na ordem certa ----------------------------------------
# A ordem desta lista importa: com `make` rodando em serie (o padrao, sem `-j`), os
# pre-requisitos sao processados na ordem em que aparecem aqui. `verificar` roda sempre
# logo depois de `particionar` e antes de `features` -- e o portao que nao pode ser
# pulado.
all: indexar particionar verificar features experimento metricas relatorio

# --- Caminho alternativo: ResNet18 pre-treinada (comparacao, ver src/resnet/features_resnet.py) --
# Alvos irmaos dos de cima, mesma logica de dependencia por arquivo, mas escrevendo em
# arquivos com nomes diferentes (caracteristicas_resnet.npy, predicoes_resnet.csv,
# metricas_resnet.md, ...) -- rodar isso NUNCA sobrescreve a saida do pipeline classico
# acima. Nao faz parte de `all` de proposito: o resultado citavel e verificado do TCC e
# o classico; este caminho e so para comparacao.
features_resnet: $(DADOS)/caracteristicas_resnet.npy

$(DADOS)/caracteristicas_resnet.npy: $(DADOS)/indice_particionado.csv src/resnet/features_resnet.py
	$(PYTHON) src/resnet/features_resnet.py

experimento_resnet: $(RESULTADOS)/predicoes_resnet.csv

$(RESULTADOS)/predicoes_resnet.csv: $(DADOS)/caracteristicas_resnet.npy src/resnet/experimento_resnet.py
	$(PYTHON) src/resnet/experimento_resnet.py

metricas_resnet: $(RESULTADOS)/metricas_resnet.md

$(RESULTADOS)/metricas_resnet.md: $(RESULTADOS)/predicoes_resnet.csv src/resnet/metricas_resnet.py
	$(PYTHON) src/resnet/metricas_resnet.py

# Mesma ordem-por-lista do `all` classico, mesmo motivo (verificar.py precisa rodar
# antes de qualquer extracao de caracteristicas). Reaproveita indexar/particionar/
# verificar do pipeline classico -- sao a mesma etapa de dados para os dois caminhos.
all_resnet: indexar particionar verificar features_resnet experimento_resnet metricas_resnet

# --- Limpeza -------------------------------------------------------------------
# Remove so o que o pipeline gera (dados_processados/ e resultados/) -- isso inclui a
# saida dos dois caminhos (classico e ResNet18), ja que os dois escrevem dentro dessas
# duas pastas. NUNCA toca em 01-datasets/, 00-documento/ ou 02-referencias/ -- sao dados
# brutos e o texto do TCC, nao saida deste pipeline.
clean:
	rm -rf $(DADOS) $(RESULTADOS)

# --- Protocolo C: BrEaST como validação externa (ver src/protocolo_c/experimento_c.py) --------------
# Treina no BUS-BRA inteiro (sem CV) e avalia, congelado, no BrEaST -- nunca mistura os
# dois: todo dado/resultado do BrEaST fica em dados_processados/breast/ e
# resultados/breast/, pastas separadas das do BUS-BRA. Depende de indexar/particionar/
# features/experimento do BUS-BRA já terem rodado (usa caracteristicas.npy e
# predicoes.csv do BUS-BRA como entrada -- nunca reescreve nenhum dos dois). Não faz
# parte de `all`/`all_resnet` de propósito, mesmo motivo do caminho ResNet18: roda-se
# explicitamente via `make protocolo_c`.
.PHONY: indexar_breast verificar_breast features_breast experimento_c metricas_breast \
        relatorio_breast protocolo_c

indexar_breast: $(DADOS)/breast/indice_breast.csv

$(DADOS)/breast/indice_breast.csv: src/protocolo_c/indexar_breast.py src/comum/indexar.py
	$(PYTHON) src/protocolo_c/indexar_breast.py

# Portão de auditoria do BrEaST -- mesma lógica de `verificar`: não produz saída própria,
# sempre re-executado quando pedido.
verificar_breast: $(DADOS)/breast/indice_breast.csv
	$(PYTHON) src/protocolo_c/verificar_breast.py

features_breast: $(DADOS)/breast/caracteristicas_breast.npy

$(DADOS)/breast/caracteristicas_breast.npy: $(DADOS)/breast/indice_breast.csv src/protocolo_c/features_breast.py src/comum/features.py $(DADOS)/caracteristicas.npy
	$(PYTHON) src/protocolo_c/features_breast.py

experimento_c: $(RESULTADOS)/breast/predicoes_breast.csv

$(RESULTADOS)/breast/predicoes_breast.csv: $(DADOS)/caracteristicas.npy $(DADOS)/breast/caracteristicas_breast.npy $(RESULTADOS)/predicoes.csv src/protocolo_c/experimento_c.py src/protocolo_ab/experimento.py src/protocolo_ab/metricas.py
	$(PYTHON) src/protocolo_c/experimento_c.py

metricas_breast: $(RESULTADOS)/breast/metricas_breast.md

$(RESULTADOS)/breast/metricas_breast.md: $(RESULTADOS)/breast/predicoes_breast.csv $(RESULTADOS)/predicoes.csv src/protocolo_c/metricas_breast.py src/protocolo_ab/metricas.py
	$(PYTHON) src/protocolo_c/metricas_breast.py

relatorio_breast: $(RESULTADOS)/breast/roc_breast.png

$(RESULTADOS)/breast/roc_breast.png: $(RESULTADOS)/breast/metricas_breast.md src/protocolo_c/relatorio_breast.py
	$(PYTHON) src/protocolo_c/relatorio_breast.py

# Alvo agregador: roda os cinco passos do Protocolo C em sequência (mais relatorio_breast,
# ver src/protocolo_c/relatorio_breast.py), na ordem certa. PRESSUPÕE que `indexar particionar
# features experimento` do BUS-BRA já rodaram antes (não estão listados aqui de propósito
# -- rodar `make all` antes, ou `make protocolo_c` depois de `make all`, nunca no lugar
# de `all`).
protocolo_c: indexar_breast verificar_breast features_breast experimento_c metricas_breast relatorio_breast
