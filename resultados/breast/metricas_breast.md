# Métricas do BrEaST — Protocolo C (validação externa)

Modelo treinado UMA vez no BUS-BRA inteiro (sem validação cruzada) e aplicado congelado ao BrEaST — nenhum ajuste foi feito olhando o BrEaST. Os dois limiares de decisão abaixo também são congelados: calculados só sobre o BUS-BRA (Protocolo A, unidade=paciente, 1.064 predições fora-de-fold agregadas por paciente) e aplicados sem reajuste aqui — ver `experimento_c.py`/`metricas_breast.py`.

- limiar de Youden (congelado no BUS-BRA): **0,5019**
- maior limiar com sensibilidade >= 0,90 (congelado no BUS-BRA): **0,3056**

IC95% por bootstrap (n=1000), reamostrando pacientes.

252 pacientes do BrEaST (252 pacientes distintos — 1 imagem por paciente, sem o conceito de duas vistas do BUS-BRA).

| métrica | valor | IC95% |
|---|---|---|
| AUROC | 0,751 | [0,694, 0,813] |
| AUPRC | 0,609 | [0,511, 0,720] |

Pontos de operação (limiar 0,5 fixo; Youden e sensibilidade>=0,90 CONGELADOS no BUS-BRA):

| ponto | sensibilidade | especificidade | acurácia | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| limiar 0,5 (fixo) | 0,745 [0,667, 0,830] | 0,656 [0,580, 0,731] | 0,690 [0,635, 0,750] | 73 | 101 | 53 | 25 |
| limiar de Youden (congelado no BUS-BRA) | 0,745 [0,667, 0,830] | 0,656 [0,580, 0,731] | 0,690 [0,635, 0,750] | 73 | 101 | 53 | 25 |
| sensibilidade >= 0,90 (congelado no BUS-BRA) | 0,898 [0,838, 0,957] | 0,370 [0,296, 0,450] | 0,575 [0,508, 0,639] | 88 | 57 | 97 | 10 |

## Comparação: validação interna (BUS-BRA) x validação externa (BrEaST)

AUROC do BUS-BRA recalculado aqui a partir do mesmo `resultados/predicoes.csv` (Protocolo A, unidade=paciente) — nunca copiado à mão, para nunca poder divergir do que está em `resultados/metricas.md`.

| base | AUROC | IC95% |
|---|---|---|
| BUS-BRA (interno, CV por paciente, Protocolo A) | 0,777 | [0,744, 0,805] |
| BrEaST (externo, congelado, Protocolo C) | 0,751 | [0,694, 0,813] |
| diferença (BrEaST − BUS-BRA) | -0,026 | — |

**Este é o número mais importante do Protocolo C**: uma AUROC mais baixa no BrEaST em relação ao BUS-BRA é o resultado esperado ao trocar de hospital/aparelho sem re-treinar — o tamanho exato dessa queda (ou não-queda) é reportado aqui sem maquiagem, seja ela grande ou pequena.

---

Bloco de dados de máquina (não editar à mão — mesmo padrão de `metricas.py`, para uma futura `relatorio_breast.py` poder marcar os pontos de operação na curva ROC sem reabrir `predicoes_breast.csv`):

<!-- DADOS_MAQUINA: {"C|paciente": {"0.5": {"fpr": 0.3441558441558441, "tpr": 0.7448979591836735}, "youden": {"fpr": 0.3441558441558441, "tpr": 0.7448979591836735}, "sens90": {"fpr": 0.6298701298701299, "tpr": 0.8979591836734694}}} -->
