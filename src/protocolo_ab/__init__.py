"""Protocolos A e B, produzidos juntos pelos mesmos scripts (ver `experimento.py`):

    A — validacao cruzada pelos 5 folds oficiais do BUS-BRA, agrupados por paciente
        (resultado principal do TCC), com unidades imagem e paciente;
    B — os mesmos dados particionados por imagem com StratifiedKFold aleatorio,
        ignorando o paciente de proposito, so para medir o tamanho do vazamento.

Os dois saem no mesmo `resultados/predicoes.csv` e sao medidos no mesmo
`resultados/metricas.md`, distinguidos pela coluna/secao `protocolo` — por isso moram
no mesmo pacote, e nao em `protocolo_a/` e `protocolo_b/` separados.
"""
