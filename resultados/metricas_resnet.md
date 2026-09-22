# Métricas do experimento — caminho ResNet18 (comparação)

Características extraídas por `features_resnet.py` (ResNet18 pré-treinada em ImageNet, penúltima camada, 512-d) em vez das características clássicas de `features.py` -- ver `resultados/metricas.md` para o resultado clássico (scikit-image + regressão logística). MESMO classificador, MESMOS folds, MESMA semente nos dois caminhos; só a extração de características muda.

IC95% por bootstrap (n=1000), reamostrando pacientes — ver `metricas_resnet.py` (cópia adaptada de `metricas.py`) para o método completo.

## Protocolo A — unidade: imagem

1875 linhas, 1064 pacientes distintos.

| métrica | valor | IC95% |
|---|---|---|
| AUROC | 0,713 | [0,686, 0,741] |
| AUPRC | 0,522 | [0,471, 0,570] |

Pontos de operação (limiares Youden e sensibilidade>=0,90 escolhidos fora de fold — ver docstring de metricas.py):

| ponto | sensibilidade | especificidade | acurácia | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| limiar 0,5 (fixo) | 0,595 [0,553, 0,638] | 0,724 [0,696, 0,750] | 0,682 [0,658, 0,705] | 361 | 918 | 350 | 246 |
| limiar de Youden | 0,679 [0,638, 0,718] | 0,660 [0,633, 0,687] | 0,666 [0,643, 0,689] | 412 | 837 | 431 | 195 |
| sensibilidade >= 0,90 | 0,900 [0,874, 0,926] | 0,302 [0,275, 0,331] | 0,495 [0,469, 0,521] | 546 | 383 | 885 | 61 |

## Protocolo A — unidade: paciente

1064 linhas, 1064 pacientes distintos.

| métrica | valor | IC95% |
|---|---|---|
| AUROC | 0,744 | [0,714, 0,773] |
| AUPRC | 0,561 | [0,504, 0,618] |

Pontos de operação (limiares Youden e sensibilidade>=0,90 escolhidos fora de fold — ver docstring de metricas.py):

| ponto | sensibilidade | especificidade | acurácia | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| limiar 0,5 (fixo) | 0,599 [0,548, 0,655] | 0,749 [0,717, 0,779] | 0,701 [0,672, 0,728] | 205 | 541 | 181 | 137 |
| limiar de Youden | 0,737 [0,687, 0,782] | 0,619 [0,581, 0,655] | 0,657 [0,627, 0,687] | 252 | 447 | 275 | 90 |
| sensibilidade >= 0,90 | 0,892 [0,858, 0,922] | 0,373 [0,339, 0,410] | 0,539 [0,510, 0,570] | 305 | 269 | 453 | 37 |

## Protocolo B — unidade: imagem

1875 linhas, 1064 pacientes distintos.

| métrica | valor | IC95% |
|---|---|---|
| AUROC | 0,723 | [0,699, 0,751] |
| AUPRC | 0,536 | [0,488, 0,589] |

Pontos de operação (limiares Youden e sensibilidade>=0,90 escolhidos fora de fold — ver docstring de metricas.py):

| ponto | sensibilidade | especificidade | acurácia | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| limiar 0,5 (fixo) | 0,575 [0,531, 0,616] | 0,729 [0,703, 0,755] | 0,679 [0,656, 0,703] | 349 | 924 | 344 | 258 |
| limiar de Youden | 0,756 [0,723, 0,793] | 0,565 [0,536, 0,595] | 0,627 [0,603, 0,652] | 459 | 716 | 552 | 148 |
| sensibilidade >= 0,90 | 0,898 [0,874, 0,923] | 0,338 [0,310, 0,367] | 0,519 [0,493, 0,546] | 545 | 428 | 840 | 62 |

---

Bloco de dados de máquina (não editar à mão; mesmo formato de `metricas.py`, reservado para uso futuro por um `relatorio_resnet.py`, que não existe ainda):

<!-- DADOS_MAQUINA: {"A|imagem": {"0.5": {"fpr": 0.2760252365930599, "tpr": 0.5947281713344317}, "youden": {"fpr": 0.33990536277602523, "tpr": 0.6787479406919276}, "sens90": {"fpr": 0.6979495268138801, "tpr": 0.899505766062603}}, "A|paciente": {"0.5": {"fpr": 0.2506925207756233, "tpr": 0.5994152046783626}, "youden": {"fpr": 0.3808864265927978, "tpr": 0.7368421052631579}, "sens90": {"fpr": 0.6274238227146814, "tpr": 0.8918128654970761}}, "B|imagem": {"0.5": {"fpr": 0.27129337539432175, "tpr": 0.5749588138385503}, "youden": {"fpr": 0.43533123028391163, "tpr": 0.7561779242174629}, "sens90": {"fpr": 0.6624605678233438, "tpr": 0.8978583196046128}}} -->
