"""Protocolo C — validação externa: treina no BUS-BRA inteiro (sem validação cruzada) e
avalia o modelo congelado no BrEaST, que ele nunca viu.

Importa de `comum/` (esquema do índice, extração de características) e de
`protocolo_ab/` (arquitetura do classificador, funções de limiar) em vez de duplicar
essa lógica — é o que garante que a comparação interno x externo mede a mesma coisa nas
duas bases. Nunca escreve em arquivo do BUS-BRA: toda saída fica em
`dados_processados/breast/` e `resultados/breast/`.
"""
