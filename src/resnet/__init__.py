"""Trilha alternativa de extracao de caracteristicas: ResNet18 pre-treinada em ImageNet,
congelada, usada so como extrator (512 caracteristicas por imagem).

NAO e um quarto protocolo: roda os mesmos Protocolos A e B de `protocolo_ab/`, trocando
so a origem das caracteristicas. Existe para comparacao e escreve em arquivos de nome
proprio (`*_resnet.*`), nunca por cima do resultado classico.
"""
