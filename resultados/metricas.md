# Métricas do experimento

IC95% por bootstrap (n=1000), reamostrando pacientes — ver `metricas.py` para o método completo.

## Protocolo A — unidade: imagem

1875 linhas, 1064 pacientes distintos.

| métrica | valor | IC95% |
|---|---|---|
| AUROC | 0,762 | [0,731, 0,792] |
| AUPRC | 0,620 | [0,565, 0,673] |

Pontos de operação (limiares Youden e sensibilidade>=0,90 escolhidos fora de fold — ver docstring do módulo):

| ponto | sensibilidade | especificidade | acurácia | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| limiar 0,5 (fixo) | 0,675 [0,629, 0,720] | 0,721 [0,691, 0,748] | 0,706 [0,680, 0,730] | 410 | 914 | 354 | 197 |
| limiar de Youden | 0,712 [0,668, 0,754] | 0,697 [0,667, 0,726] | 0,702 [0,676, 0,725] | 432 | 884 | 384 | 175 |
| sensibilidade >= 0,90 | 0,900 [0,872, 0,928] | 0,362 [0,332, 0,393] | 0,536 [0,509, 0,564] | 546 | 459 | 809 | 61 |

## Protocolo A — unidade: paciente

1064 linhas, 1064 pacientes distintos.

| métrica | valor | IC95% |
|---|---|---|
| AUROC | 0,777 | [0,744, 0,805] |
| AUPRC | 0,638 | [0,585, 0,692] |

Pontos de operação (limiares Youden e sensibilidade>=0,90 escolhidos fora de fold — ver docstring do módulo):

| ponto | sensibilidade | especificidade | acurácia | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| limiar 0,5 (fixo) | 0,705 [0,657, 0,753] | 0,733 [0,700, 0,763] | 0,724 [0,695, 0,749] | 241 | 529 | 193 | 101 |
| limiar de Youden | 0,678 [0,631, 0,729] | 0,749 [0,717, 0,780] | 0,727 [0,698, 0,752] | 232 | 541 | 181 | 110 |
| sensibilidade >= 0,90 | 0,892 [0,857, 0,925] | 0,425 [0,389, 0,464] | 0,575 [0,545, 0,607] | 305 | 307 | 415 | 37 |

## Protocolo B — unidade: imagem

1875 linhas, 1064 pacientes distintos.

| métrica | valor | IC95% |
|---|---|---|
| AUROC | 0,782 | [0,753, 0,810] |
| AUPRC | 0,648 | [0,596, 0,697] |

Pontos de operação (limiares Youden e sensibilidade>=0,90 escolhidos fora de fold — ver docstring do módulo):

| ponto | sensibilidade | especificidade | acurácia | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| limiar 0,5 (fixo) | 0,718 [0,676, 0,762] | 0,714 [0,682, 0,742] | 0,715 [0,691, 0,738] | 436 | 905 | 363 | 171 |
| limiar de Youden | 0,666 [0,621, 0,710] | 0,760 [0,733, 0,786] | 0,730 [0,705, 0,752] | 404 | 964 | 304 | 203 |
| sensibilidade >= 0,90 | 0,904 [0,875, 0,932] | 0,414 [0,383, 0,445] | 0,573 [0,545, 0,601] | 549 | 525 | 743 | 58 |

---

Bloco de dados de máquina para `relatorio.py` (não editar à mão; ver docstring de `metricas.py`, seção "Bloco de dados de máquina"):

<!-- DADOS_MAQUINA: {"A|imagem": {"0.5": {"fpr": 0.2791798107255521, "tpr": 0.6754530477759473}, "youden": {"fpr": 0.3028391167192429, "tpr": 0.7116968698517299}, "sens90": {"fpr": 0.63801261829653, "tpr": 0.899505766062603}}, "A|paciente": {"0.5": {"fpr": 0.2673130193905817, "tpr": 0.7046783625730995}, "youden": {"fpr": 0.2506925207756233, "tpr": 0.6783625730994152}, "sens90": {"fpr": 0.574792243767313, "tpr": 0.8918128654970761}}, "B|imagem": {"0.5": {"fpr": 0.2862776025236593, "tpr": 0.7182866556836903}, "youden": {"fpr": 0.23974763406940058, "tpr": 0.6655683690280065}, "sens90": {"fpr": 0.5859621451104101, "tpr": 0.9044481054365733}}} -->
