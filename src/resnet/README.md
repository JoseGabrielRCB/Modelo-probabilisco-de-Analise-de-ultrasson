# Dir RESNET

**Contexto**: Caminho alternativo de extração de caracteristicas, usando uma ResNet18 pré-treinada no ImageNet, congelada e usada só como extrator (sem nenhum treino). NÃO é um quarto protocolo, roda os mesmos Protocolos A e B do `protocolo_ab/` trocando só de onde vem as caracteristicas. Existe pra comparar com o caminho classico, o resultado citavel do TCC continua sendo o classico

## Base comum a todos arquivos

Todos seguem o seguinte padrão:
**Arquivos proprios** : tudo que sai daqui tem `_resnet` no nome (`caracteristicas_resnet.npy`, `predicoes_resnet.csv`, `metricas_resnet.md`...), nunca escreve por cima do resultado classico
**Mesma base** : mesmo `indice_particionado.csv`, mesmos folds, mesmo classificador e mesma semente, só a extração muda

## Arquivos

## features_resnet.py - Etapa 4

Lê o `indice_particionado.csv` e gera um vetor de **512 caracteristicas** por imagem

Pra cada imagem:

1. converte pra RGB (o cinza é replicado em R=G=B)
2. redimensiona pra 224x224 (entrada padrão da ResNet)
3. escala pra [0,1] e normaliza com media/desvio do ImageNet
4. passa pela ResNet18 com a `fc` trocada por `Identity`, o vetor é a saida do avgpool

- carregar_modelo() - pesos `IMAGENET1K_V1`, em `.eval()` e com todos os pesos congelados (Nao se alteram)
- preprocessar_imagem() - faz os passos 1 a 3
- extrair_todas() - roda em lotes de 32 (só por velocidade na CPU, não muda o resultado) dentro do `torch.no_grad()`

Saidas: `caracteristicas_resnet.npy` (N x 512, float64) e `caracteristicas_resnet_ids.csv` com as mesmas colunas do classico (`id, paciente, fold, rotulo, base, imagem`)

## experimento_resnet.py - Etapa 5

Chama o `main()` do `protocolo_ab/experimento.py` só trocando os arquivos de entrada/saida. Grava `resultados/predicoes_resnet.csv` sem tocar no `predicoes.csv`

## metricas_resnet.py - Etapa 6

Chama o `main()` do `protocolo_ab/metricas.py` (bootstrap por paciente, limiares fora de fold, alerta de sanidade, tudo igual), só com os textos do cabeçalho proprios da ResNet. Grava `metricas_resnet.md` e `roc_resnet.csv`

**Nota** : Não tem relatorio (figura) pra ResNet ainda. Não faz parte do `make all` de proposito, roda separado com `make all_resnet`.
