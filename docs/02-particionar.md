# `particionar.py` — especificação da Etapa 2

**Status:** a escrever (pelo autor). Pré-requisito: `dados_processados/indice.csv` já existe e
passou nas verificações da Etapa 1 (✅ confirmado: 1.875 linhas, 1.064 pacientes, 1.268/607).

**Escopo desta versão:** atribuir cada imagem do BUS-BRA a um dos 5 folds **oficiais** do dataset
(decisão já registrada no projeto: usar os folds dos autores, não criar uma partição própria — dá
comparabilidade direta com o baseline publicado, e uma etapa de implementação a menos). O que fica
para depois: o split "por imagem" de propósito (Protocolo B, o experimento de vazamento) e a
partição do BrEaST — nenhum dos dois entra neste script agora.

---

## 1. O contrato do script

**Lê:**
- `dados_processados/indice.csv` (saída da Etapa 1)
- `01-datasets/BUS-BRA/BUSBRA/5-fold-cv.csv` (arquivo oficial, distribuído pelos autores do
  dataset — só leitura, nunca editado)

**Escreve:** `dados_processados/indice_particionado.csv` — o mesmo `indice.csv`, mais uma coluna
`fold`.

**Regra de fronteira:** a partir desta etapa, nenhum script seguinte (`verificar.py`,
`features.py`, `experimento.py`) lê `5-fold-cv.csv` diretamente — todos leem só
`indice_particionado.csv`.

---

## 2. Um detalhe que faltou na Etapa 1: falta um identificador por imagem

O `indice.csv` de hoje tem a coluna `paciente` (o número do `Case`, ex.: `1`), mas não tem uma
coluna com o identificador **da imagem** (ex.: `bus_0001-l`). O arquivo oficial de folds é indexado
por esse identificador — então esta etapa precisa dele para poder juntar as duas tabelas.

Duas formas de resolver, escolha uma:

**Opção A — extrair do caminho da imagem, dentro do próprio `particionar.py`.** O nome do arquivo,
sem a extensão, já é o identificador (`Images/bus_0001-l.png` → `bus_0001-l`). Resolve sem tocar no
`indexar.py`, mas é um pouco frágil: qualquer mudança futura na convenção de nome de arquivo quebra
essa extração silenciosamente.

**Opção B — voltar ao `indexar.py` e adicionar uma coluna `id` com esse valor, direto da coluna
`ID` do `bus_data.csv`** (que já existe lá, ex.: `bus_0001-l` — é literalmente a mesma coluna que
virou a base do `paciente` e dos caminhos de arquivo, só que sem processar). Mais robusto, e o
`id` também é uma das colunas exigidas depois no `predicoes.csv` (Etapa 5), então esse trabalho não
é perdido.

**Recomendação:** Opção B — é uma mudança pequena no `indexar.py` (uma coluna a mais, direto da
tabela original) e evita que o mesmo problema reapareça na Etapa 5.

---

## 3. O arquivo oficial de folds, exatamente como é (conferido, não suposto)

`5-fold-cv.csv` — 1.875 linhas, 8 colunas: `ID, Pathology, kFold, valid_1, valid_2, valid_3,
valid_4, valid_5`.

Exemplo de linha real:
```
ID=bus_0001-l, Pathology=malignant, kFold=3, valid_1=1.0, valid_2=1.0, valid_3=NaN, valid_4=1.0, valid_5=1.0
```

O que interessa aqui é só `ID` e `kFold` — a coluna `kFold` já diz, para cada imagem, a qual dos 5
folds ela pertence (valores de 1 a 5). As colunas `valid_1`...`valid_5` parecem ser máscaras
auxiliares dos autores para outro propósito (validação cruzada aninhada, possivelmente) — **não
são necessárias para o protocolo deste TCC** e podem ser ignoradas.

---

## 4. O algoritmo, passo a passo

1. **Ler `indice.csv`** (saída da Etapa 1).

2. **Garantir a coluna `id`** — seguindo a Opção B da seção 2, ou a extração da Opção A.

3. **Ler `5-fold-cv.csv`**, mantendo só as colunas `ID` e `kFold`.

4. **Juntar as duas tabelas** (`pandas.merge`, tipo `left`, casando `id` de um lado com `ID` do
   outro). Renomear `kFold` para `fold` no resultado.

5. **Conferir que a junção não perdeu nem duplicou nenhuma linha:** o número de linhas depois do
   merge tem que continuar sendo 1.875, e a coluna `fold` não pode ter nenhum valor vazio. Se
   sobrar `fold` vazio, significa que algum `id` do seu índice não bateu com nenhum `ID` do arquivo
   oficial — isso é motivo de parar e investigar, não de seguir em frente com dado faltando.

6. **A verificação central desta etapa (o "gate"):** para cada `paciente`, confira que **todos os
   folds atribuídos às imagens dele são o mesmo valor**. Em código isso é: agrupar por `paciente`,
   contar quantos valores distintos de `fold` aparecem em cada grupo, e travar a execução se algum
   grupo tiver mais de um valor distinto. Essa é a verificação que transforma "usei o arquivo que
   veio junto" em "auditei o arquivo que veio junto" — é uma frase que vai direto para a seção de
   Materiais e Métodos do TCC.

7. **Imprimir o resumo:** quantas imagens caíram em cada fold, e a confirmação de que zero
   pacientes aparecem em mais de um fold.

8. **Salvar** em `dados_processados/indice_particionado.csv`.

---

## 5. Critério de "pronto"

Já testei este passo de ponta a ponta como conferência (sem escrever o script) para você saber
exatamente o que esperar:

- Linhas depois do merge: **1.875** (nenhuma perdida, nenhuma duplicada).
- Contagem de imagens por fold: **376, 385, 366, 365, 383** (folds 1 a 5, nessa ordem).
- Pacientes com mais de um fold distinto: **0**.

Se o seu script chegar a esses mesmos números, a Etapa 2 está correta. Se algum número diferir,
pare antes de seguir para a Etapa 3 — o problema mais provável é a junção da seção 4 (verifique se
o `id` que você construiu bate exatamente com o formato do `ID` do arquivo oficial, sem espaço ou
maiúscula/minúscula diferente).

---

## 6. O que fica para depois (não fazer agora)

- **Protocolo B (split por imagem, de propósito):** por enquanto o único modo de partição é o
  oficial. O split aleatório por imagem — que existe só para *medir* o efeito do vazamento — entra
  mais para a frente, provavelmente dentro do `experimento.py`, como uma segunda forma de rodar o
  mesmo treino, não como um novo modo deste script.
- **Partição do BrEaST:** o BrEaST inteiro é reservado para validação externa (Protocolo C) — ele
  nunca entra em nenhum fold de treino/teste do BUS-BRA. Quando a hora chegar, ele nem passa por
  este script: o modelo treinado no BUS-BRA é aplicado direto nele, sem re-particionar nada.
