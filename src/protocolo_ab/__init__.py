"""Protocolos A e B, produzidos juntos pelos mesmos scripts (ver `experimento.py`):

    A — validação cruzada pelos 5 folds oficiais do BUS-BRA, agrupados por paciente
        (resultado principal do TCC), com unidades imagem e paciente;
    B — os mesmos dados particionados por imagem com StratifiedKFold aleatório,
        ignorando o paciente de propósito, só para medir o tamanho do vazamento.

Os dois saem no mesmo `resultados/predicoes.csv` e são medidos no mesmo
`resultados/metricas.md`, distinguidos pela coluna/seção `protocolo` — por isso moram
no mesmo pacote, e não em `protocolo_a/` e `protocolo_b/` separados.
"""
