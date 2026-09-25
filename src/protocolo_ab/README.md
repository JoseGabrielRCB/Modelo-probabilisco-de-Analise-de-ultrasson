# Dir PROTOCOLO_AB

**Contexto**: Aqui ficam os Protocolos A e B juntos, porque saem dos mesmos scripts e vão pro mesmo `predicoes.csv` e `metricas.md` (separados pela coluna `protocolo`). O A é o resultado principal do TCC, validação cruzada com os 5 folds oficiais agrupados por paciente. O B existe só pra medir o vazamento, mesmo dado mas dividido por imagem ignorando o paciente de proposito. Comparar A x B mostra o quanto o vazamento infla o resultado

## Base comum a todos arquivos

Todos seguem o seguinte padrão:
**exigir(condicao,mensagem)** : importado do `comum/indexar.py`, se qualquer checagem falha para a execucao antes de gravar
**Reaproveitamento** : `resnet/` e `protocolo_c/` importam as funcoes daqui em vez de copiar, assim o classificador e as metricas são exatamente os mesmos em todos os caminhos

## Arquivos

## experimento.py - Etapa 5

Lê `caracteristicas.npy` e `caracteristicas_ids.csv` (saida do features.py) e grava `resultados/predicoes.csv` com as colunas (`protocolo,unidade,dataset,id,paciente,fold,y_true,y_score`). classificador é StandardScaler + LogisticRegression (`max_iter=2000`, `class_weight="balanced"`), semente 42

- novo_pipeline() - cria um pipeline novo a cada fold, **nunca reaproveitar um treinado** senão o scaler vê dado de teste e vaza entre folds
- rodar_protocolo_a() - treina fold a fold pelos 5 folds oficiais, gera predição por imagem e depois agrega por paciente (media das probabilidades das imagens dele), por isso o A tem as unidades `imagem` e `paciente`
- rodar_protocolo_b() - StratifiedKFold aleatorio por imagem (5 folds, semente 42), só unidade `imagem`, o vazamento aqui é proposital
- resumir() - impime quantas predições e malignas saiu em cada protocolo/unidade

## metricas.py - Etapa 6

Calcula as metricas finais a partir do `predicoes.csv` pra cada combinação (A/imagem, A/paciente, B/imagem). Grava `resultados/metricas.md` e `resultados/roc.csv`

- AUROC e AUPRC com IC95% por bootstrap (1000 repetições)
- 3 pontos de operação: limiar fixo 0,5 , Youden e maior limiar com sensibilidade >= 0,90. Em cada um sai sensibilidade, especificidade, acuracia e matriz de confusão (TP,TN,FP,FN), todos com IC95%

- limiares_fora_de_fold() - **Parte central da metodologia** o limiar usado nas linhas do fold f é escolhido só com as predições dos OUTROS folds, nenhuma linha é decidida por um limiar qe viu ela. Por isso o limiar muda um pouco de fold pra fsold
- gerar_reamostragens_por_paciente() - o bootstrap sorteia PACIENTES e não imagens, porque imagens do mesmo paciente não são independentes
- ic95_bootstrap() - percentis 2,5 e 97,5 , descarta amostra que ficou com uma classe só e trava se descartar mais de 10%
- metricas_de_um_combo() / metricas_nos_pontos() - juntam tudo isso pra um combo protocolo+unidade
- verificar_criterio_de_sanidade() - se a acuracia do A/paciente 0,5 passar de 0,95 só avisa (não trava)
- montar_markdown() - escreve o `.md`, no final vai um bloco `DADOS_MAQUINA` (json dentro de comentario html) com os pontos (fpr,tpr) que o relatorio.py lê

## relatorio.py - Etapa 7

Desenha a curva ROC do Protocolo A por paciente em `resultados/curva_roc.png`. Não recalcula nada, só lê o `metricas.md` (bloco DADOS_MAQUINA) e o `roc.csv`

- desenhar_roc() - desenha a curva e marca o s3 pontos de operação, o `protocolo_c/relatorio_breast.py` usa a mesma função pro BrEaST

**Nota** : A ordem é experimento -> metricas -> relatorio, cada um depende da saida do anterior. O makefile já roda nessa ordem (`make experimento metricas relatorio`)
