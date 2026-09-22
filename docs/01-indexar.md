# `indexar.py` — especificação da Etapa 1

**Status:** a escrever (pelo autor). Este documento descreve o que o script precisa fazer e por
quê, sem código pronto — é a referência para escrever e revisar `src/indexar.py`.

**Escopo desta versão:** só o BUS-BRA. O BrEaST entra depois como uma segunda função de leitura
dentro do mesmo script (ver seção "Extensão futura: BrEaST" no final) — o resto do pipeline não
muda quando isso acontecer.

---

## 1. O contrato do script

**Lê:** só os arquivos brutos do BUS-BRA — `01-datasets/BUS-BRA/BUSBRA/bus_data.csv`, a pasta
`Images/`, a pasta `Masks/`. Nunca escreve nada dentro dessas pastas.

**Escreve:** um único arquivo, `dados_processados/indice.csv`.

**Regra de fronteira do projeto, a partir de agora:** nenhum outro script (`particionar.py`,
`features.py`, `experimento.py`...) abre `bus_data.csv` de novo, nem olha dentro de `Images/`
diretamente. Todos leem só o `indice.csv`. Isso garante que, se um número aparecer errado depois,
o bug está ou aqui (montagem do índice) ou lá na frente — nunca nos dois ao mesmo tempo, porque só
um lugar toca os dados brutos.

---

## 2. Esquema exato da tabela de saída

Uma linha por imagem, com estas colunas:

| coluna | tipo | de onde vem | observação |
|---|---|---|---|
| `base` | texto | fixo `"bus-bra"` | prepara o terreno para quando o BrEaST entrar com `"breast"` |
| `paciente` | número | coluna `Case` | vira o identificador usado no split por paciente (Etapa 2) |
| `imagem` | texto (caminho) | construído a partir de `ID` | caminho do PNG, relativo à raiz do projeto |
| `mascara` | texto (caminho) | construído a partir de `ID` | ver a pegadinha na seção 4 — não é cópia direta do nome da imagem |
| `rotulo` | 0 ou 1 | coluna `Pathology` traduzida | 1 = malignant, 0 = benign |
| `birads` | texto | coluna `BIRADS` | guardar como **texto**, não número — no BrEaST isso vem misturado (`"4a"`, `"4b"`...), então tratar os dois como texto desde já evita remendo depois |
| `aparelho` | texto | coluna `Device` | para a análise por subgrupo, mais pra frente |
| `largura`, `altura` | número | `Width`, `Height` | |
| `lado` | texto | `Side` | `left` / `right` / `single` — não decide nada aqui, só carrega a informação para a agregação por paciente, depois |
| `sha1` | texto | calculado no script | impressão digital do conteúdo do arquivo — serve para achar duplicatas depois, de graça |

---

## 3. Os dois arquivos brutos, exatamente como são (conferidos, não supostos)

**`bus_data.csv`** — 1.875 linhas, 10 colunas: `ID, Case, Histology, Pathology, BIRADS, Device,
Width, Height, Side, BBOX`.

Exemplo de linha real:
```
ID=bus_0001-l, Case=1, Histology=invasive ductal carcinoma, Pathology=malignant,
BIRADS=4, Device=GE Logiq 7 @10-14MHz, Width=274, Height=353, Side=left, BBOX=[91,24,103,79]
```

`Pathology` só tem dois valores possíveis no arquivo: `malignant` e `benign`. Sem ambiguidade,
sem valor faltando.

---

## 4. A pegadinha do nome do arquivo de máscara

As imagens seguem o `ID` da tabela direto:

```
ID = bus_0001-l   →   arquivo = Images/bus_0001-l.png
```

As máscaras **não** seguem o mesmo padrão — o prefixo `bus_` é trocado por `mask_`:

```
ID = bus_0001-l   →   arquivo = Masks/mask_0001-l.png
```

Se o script simplesmente reusar o `ID` para montar o nome do arquivo de máscara
(`Masks/{ID}.png`), ele vai procurar um arquivo que não existe. A transformação certa é: pegar o
`ID`, tirar o `bus_` do começo, colocar `mask_` no lugar. Em código isso é uma linha — por exemplo:

```python
nome_mascara = "mask_" + ID.removeprefix("bus_")
```

— mas o ponto não é a linha em si, é que essa suposição precisa ser **verificada no exemplo real**,
não adivinhada. É exatamente o tipo de erro que passa despercebido e vira um bug silencioso: a
máscara de um paciente sendo associada, por acidente, à imagem de outro.

---

## 5. O algoritmo, passo a passo

1. **Ler `bus_data.csv`** com `pandas.read_csv`. Usar caminho relativo à raiz do projeto (via
   `pathlib`), não um caminho absoluto do Windows escrito na mão — assim o script continua
   funcionando se a pasta for movida ou rodar em outra máquina.

2. **Montar o caminho da imagem** para cada linha, usando o `ID` direto (seção 3).

3. **Montar o caminho da máscara**, aplicando a troca de prefixo (seção 4).

4. **Conferir que os dois arquivos existem de verdade no disco** antes de aceitar a linha. Se um
   caminho não existir, o script deve **parar com erro**, apontando qual `ID` falhou — não pular a
   linha silenciosamente e seguir em frente. Parece exagero para 1.875 imagens que já sabemos que
   batem, mas é o hábito que vale a pena pegar agora: é o mesmo princípio do `verificar.py` da
   Etapa 3, e o TCC inteiro é sobre como esse tipo de verificação costuma faltar na área.

5. **Traduzir o rótulo:** `Pathology == 'malignant'` vira `1`, `Pathology == 'benign'` vira `0`.
   Além de traduzir, o script deve **conferir que só esses dois valores aparecem** na coluna e
   travar se aparecer um terceiro. Parece redundante agora, mas essa mesma função de leitura vira
   o modelo para a função do BrEaST — lá existe um terceiro valor (`normal`), e essa verificação é
   o que obriga a decidir explicitamente o que fazer com ele, em vez de deixá-lo vazar para dentro
   do treino sem querer.

6. **Calcular o SHA-1 de cada imagem:** abrir o arquivo em modo binário, ler os bytes, passar pela
   função de hash (`hashlib.sha1`). Rápido — não deve passar de poucos segundos para as 1.875
   imagens. Não é sobre segurança: é uma "impressão digital" do conteúdo — dois arquivos com o
   mesmo hash são byte-a-byte idênticos, o que permite descobrir depois, sem esforço, se alguma
   imagem está duplicada.

7. **Renomear as colunas** para o esquema da seção 2 (`Case` → `paciente` etc.) e **adicionar a
   coluna `base`** fixa com o valor `"bus-bra"`.

8. **Rodar as verificações de sanidade** antes de salvar qualquer coisa:
   - o número de linhas bate com 1.875?
   - o número de valores distintos em `paciente` bate com 1.064?
   - nenhum `sha1` está duplicado?
   - nenhuma coluna importante ficou vazia (`null`)?

   Se qualquer uma dessas falhar, o script deve avisar e parar — nunca salvar um `indice.csv`
   quebrado silenciosamente.

9. **Salvar** o resultado em `dados_processados/indice.csv` (sem a coluna de índice do pandas).

10. **Imprimir um resumo** no final: quantidade de imagens, de pacientes, proporção de malignos.
    Não é só para debug — essa contagem (algo como *"BUS-BRA: 1.875 imagens, 1.064 pacientes, 607
    malignas, 32,4%"*) é o texto que entra quase pronto na seção de Materiais e Métodos, como
    evidência medida, não apenas citada da literatura.

---

## 6. Princípios de desenho (o porquê da separação em etapas)

- **Fonte única de verdade:** depois desta etapa, ninguém mais lê os arquivos brutos.
- **Idempotência:** rodar o script duas vezes produz o mesmo `indice.csv` — sem efeito colateral
  acumulado.
- **Auditável a olho nu:** `indice.csv` é uma tabela simples, abre em qualquer editor ou Excel.
- **Só leitura nos dados brutos:** protege contra corromper os dados originais por engano.

---

## 7. Critério de "pronto"

- O script roda do início ao fim sem erro.
- `indice.csv` tem 1.875 linhas e 1.064 valores distintos em `paciente`.
- A primeira linha, ao abrir com `pandas.read_csv(...).head()`, bate com o exemplo da seção 3:
  paciente 1, maligno, BIRADS 4.

---

## 8. Extensão futura: BrEaST (não fazer agora)

Quando o protocolo A (só BUS-BRA) estiver funcionando ponta a ponta, a extensão para o BrEaST é:
uma segunda função de leitura, só para o arquivo
`BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx`, que devolve uma tabela no **mesmo esquema** da
seção 2. Pontos de atenção já mapeados para quando chegar a hora:

- O identificador de paciente lá é `CaseID`, não `Case`.
- O rótulo está em `Classification`, com **três** valores (`benign`, `malignant`, `normal`) — os
  `normal` (4 casos) precisam ser descartados para a tarefa binária; dá para identificá-los também
  pela coluna `Mask_tumor_filename` vazia.
- `BIRADS` já vem como texto misturado (`"4a"`, `"4b"`, `"4c"`...) — por isso a coluna `birads` no
  esquema comum (seção 2) já foi definida como texto, e não número, desde o BUS-BRA.
- Os nomes dos arquivos de imagem e máscara já vêm prontos nas colunas `Image_filename` e
  `Mask_tumor_filename` — não precisa reconstruir caminho por convenção como no BUS-BRA.

O resto do script (cálculo de hash, verificações de sanidade, formato de saída) é reaproveitado
sem mudança — só a função de leitura muda.
