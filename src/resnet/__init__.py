"""Trilha alternativa de extração de características: ResNet18 pré-treinada em ImageNet,
congelada, usada só como extrator (512 características por imagem).

NÃO é um quarto protocolo: roda os mesmos Protocolos A e B de `protocolo_ab/`, trocando
só a origem das características. Existe para comparação e escreve em arquivos de nome
próprio (`*_resnet.*`), nunca por cima do resultado clássico.
"""
