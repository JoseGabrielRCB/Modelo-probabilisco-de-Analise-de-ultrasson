# Dir COMUM

**Contexto**: Neste Dir foi reunido todos os arquviso e funcionalidades correspontes que são utilizados por todos os outros protocolos (incluindo Resnet).Base da metodologia, que garante o não vazamento de dados e processamento correto, lendo os dados brutos e convertendo em arquivos padronizados para or estante

## Base comum a todos arquivos

Todos seguem o seguinte padrão:
**exigir(condicao,mensagem)** : Para qualquer execucao quando qualquer checagem falha, se ocorrer para a gravação em disco
**verificar_sanidade(...)** : Confere contagens esperadas, duplicatas e valores antes de gravar qualquer saida

## Arquivos

## indexar.py - Etapa 1

Realiza a leitura dos arquivos brutos do BUS-BRA e grava no esquema comum 'COLUNAS'
(`base,id,paciente,imagem,mascara,rotulo,birads,aparelo,largura,altura,lado,sha`), pegar todos esses dados mesmo que não usemos todos é de suma importancia, foi a partir dessa analise e de não presumir padrnização correta dos dados, que encontramos os erro dos DataSet

- ler_bus_bra() : monta os cminhos de imagem  e mascra a partir do ID, confirma que todso estão corretamente salvos , aceita penas `binigno == 0` `malignat== 1`
- sha1_do_arquivo - calcula um id para cada imagem, usada para dectar arquivos com conteudos identicos
-verificar_sanidade = exige os valores brutos de imagem,paciente , nenhum dos sha1 repete ou tem coluna vazia

## particiona.py - Etapa 2

Tem como função juntar indice.csv ao arquivo 5-fold-cv.csv do BUS-BRA e slava em `dados_processadors/indice_particionado.csv` com a coluna `fold` , usando o valor dos autores para manter a comparabilidade. o slipt por imagem do Protocolo B não fica aqui.

-ler_fold_oficias - mantém apenas as colunas `ID` e `kfold`

- juntar() - faz a mesclagem pelo id e grante que não foi perdida nenhuma das linhas, se houve douplicação ou ficou sem o `fold`
- verificar_gate_por_paciente() - **Parte central da metodologia** todas as imagem de paciente precisam estar no mesmo `fold`, para evitar vazemtno entre treino e teste
-verificar_sanidade() - confere o total de linha se contagem exata por `fold`

### verificar.py - Etapa 3

Veirfica `indice_particionado.csv` contra os arquivos reais salvos.

-verificar_mpaciente_unico_por_fold - refaz verificação por paciente
-verificar_sha1_sem_duplicata() - nenhuma imagem repete

- verificar_mascaras - toda imagem tem mascara e ambas tem as mesmas dimensoes, medidas nos arquivos reais (altura declarada pelo dataset está errada em
  algumas linhas)

  ### `features.py` — Etapa 4: características clássicas

Calcula um vetor de **62 características** por imagem, a partir da imagem inteira em
tons de cinza redimensionada para 256×256 (sem recorte pela máscara).

| Função | Categoria | Qtd. |
| --- | --- | --- |
| features_intensidade() | média, desvio, mín., máx., mediana, assimetria, curtose, percentis 10/25/75/90 | 11 |
| features_glcm() | textura GLCM/Haralick (média entre 4 ângulos) | 6 |
| features_lbp() | histograma normalizado do LBP uniforme (P=8, R=1) | 10 |
| features_sobel() | média, desvio e máximo da magnitude do gradiente | 3 |
| features_grade() | média e desvio de cada bloco de uma grade 4×4 | 32 |

- extrair_vetor() — concatena as cinco categorias na ordem acima.
- extrair_todas() — percorre o índice preservando a ordem e imprime o progresso.
- verificar_sanidade() — confere número de linhas e ausência de `NaN`/`Inf`.
- resumir() — confere que a soma das categorias bate com a dimensão real da matriz.

Saídas, alinhadas linha a linha:

- `dados_processados/caracteristicas.npy` — matriz N × 62 em `float64`;
- `dados_processados/caracteristicas_ids.csv` — colunas `id, paciente, fold, rotulo, base, imagem`,
  que ligam cada linha da matriz ao paciente, ao rótulo e ao fold.

**Nota** : Os arquivos podem ser executados de forma independente, mas utilize o makefile para execulatos em sequencia, durante o processo de criação separei assim para facilitar os testes pessoais.
