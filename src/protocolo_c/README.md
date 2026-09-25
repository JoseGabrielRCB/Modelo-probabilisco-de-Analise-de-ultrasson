# Dir PROTOCOLO_C

**Contexto**: Protocolo C é a validação externa. Treina UM modelo no BUS-BRA inteiro (sem validação cruzada, a coluna `fold` nem é usada) e aplica ele congelado no BrEaST, que o modelo nunca viu. Nenhuma escolha (pesos, padronização, limiares) olha pro BrEaST. Serve pra ver o quanto o resultado cai quando troca de hospital/aparelho sem re-treinar

## Base comum a todos arquivos

Todos seguem o seguinte padrão:
**Importa em vez de copiar** : esquema de colunas e extração vem do `comum/`, classificador e metricas vem do `protocolo_ab/`. É isso que garante que a comparação interno x externo mede a mesma coisa nas duas bases
**Nunca escreve no BUS-BRA** : tudo do BrEaST fica em `dados_processados/breast/` e `resultados/breast/`, o BUS-BRA só é lido
**exigir(condicao,mensagem)** : mesmo do comum, se alguma checagem falha para antes de gravar

## Arquivos

## indexar_breast.py - Etapa 1

Lê a planilha clinica `BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx` e a pasta de imagens/mascaras, grava `dados_processados/breast/indice_breast.csv` no mesmo esquema `COLUNAS` do BUS-BRA (importado do `comum/indexar.py`, não copiado)

- ler_breast() - confere que a planilha tem 256 linhas, aceita só `benign`, `malignant` ou `normal`. **Os 4 `normal` são descartados ANTES de montar caminho de mascara** porque não tem lesão nem mascara, sobram 252
- `paciente` = `CaseID`, 1 imagem por paciente (não tem o conceito de duas vistas do BUS-BRA)
- `birads` fica como texto ("4a","4b"...), `aparelho` e `lado` ficam vazios de proposito (não existem na planilha)
- tamanho_da_imagem() - largura/altura lidas da imagem real, a planilha não declara
- verificar_sanidade() - 252 linhas, 252 pacientes, nenhum sha1 repetido, nenhuma coluna critica vazia e rotulo so 0 ou 1

## verificar_breast.py - Etapa 2

Portão de auditoria do BrEaST, igual ao `comum/verificar.py`. Lê o `indice_breast.csv` e so pra leitura o `indice.csv` do BUS-BRA

- verificar_sha1_sem_duplicata() - (a) nenhuma imagem repetida dentro do BrEaST
- verificar_mascaras() - (b) toda mascara existe e tem a mesma dimensao da imagem, os dois lidos do arquivo real
- verificar_sem_intersecao_com_busbra() - (c) **nenhum sha1 do BrEaST pode aparecer no BUS-BRA**, se a mesma imagem estiver nas duas bases ela conta como treino e teste e a validação externa perde o sentido. Se o indice do BUS-BRA ainda não existe so avisa e pula
- reportar_balanco() - (d) só informa, benignas x malignas

## features_breast.py - Etapa 3

Calcula o MESMO vetor de 62 caracteristicas do BUS-BRA, usando o `extrair_vetor()` do `comum/features.py` (mesmas categorias, mesma ordem, mesmo 256x256)

- extrair_todas() - percorre o indice na ordem e mostra o progresso a cada 50 imagens
- usa o verificar_sanidade() do comum (uma linha por imagem, sem NaN/Inf)

Saidas: `caracteristicas_breast.npy` (252 x 62) e `caracteristicas_breast_ids.csv` (`id, paciente, rotulo`), alinhados linha a linha

## experimento_c.py - Etapa 4

Treina o `novo_pipeline()` do Protocolo A com as 1875 imagens do BUS-BRA e aplica no BrEaST. Grava `resultados/breast/predicoes_breast.csv`, uma linha por paciente (`protocolo="C"`, `unidade="paciente"`, `dataset="breast"`)

## metricas_breast.py - Etapa 5

Mesmo metodo do `protocolo_ab/metricas.py` (AUROC/AUPRC, 3 pontos de operação, IC95% por bootstrap de pacientes). Grava `metricas_breast.md` e `roc_breast.csv`

- **Limiares CONGELADOS** : no BrEaST não tem fold, então o Youden e o sens>=0,90 são calculados uma vez só nas 1064 predições do A/paciente do BUS-BRA e aplicados sem reajuste nas 252 do BrEaST
- metricas_do_breast() - metricas do BrEaST com esses dois limiares congelados
- metricas_busbra_interno() - recalcula a AUROC do BUS-BRA a partir do mesmo `predicoes.csv` (nunca copiada na mão pra nunca divergir do `metricas.md`)
- montar_markdown() - a tabela de comparação interno x externo com a diferença (BrEaST - BUS-BRA) é **o numero mais importante do Protocolo C**, reportado do jeito que saiu seja a queda grande ou pequena

## relatorio_breast.py - Etapa 6

Curva ROC do BrEaST em `resultados/breast/roc_breast.png`, usando o mesmo `desenhar_roc()` do `protocolo_ab/relatorio.py`, com os pontos 0,5 / Youden / sens>=0,90 congelados

**Nota** : Depende do BUS-BRA já ter rodado antes (`make all`), porque usa o `caracteristicas.npy` e o `predicoes.csv` dele como entrada. Depois é só `make protocolo_c` que roda as 6 etapas na ordem
