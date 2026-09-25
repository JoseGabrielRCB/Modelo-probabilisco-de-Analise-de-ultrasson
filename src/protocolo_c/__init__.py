"""Protocolo C — validacao externa: treina no BUS-BRA inteiro (sem validacao cruzada) e
avalia o modelo congelado no BrEaST, que ele nunca viu.

Importa de `comum/` (esquema do indice, extracao de caracteristicas) e de
`protocolo_ab/` (arquitetura do classificador, funcoes de limiar) em vez de duplicar
essa logica — e o que garante que a comparacao interno x externo mede a mesma coisa nas
duas bases. Nunca escreve em arquivo do BUS-BRA: toda saida fica em
`dados_processados/breast/` e `resultados/breast/`.
"""
