# Modelo-probabilisco-de-Analise-de-ultrasson

Este trabalho refe-se ao TCC "Classificação de lesões mamárias em ultrassonografia:
avaliação sob particionamento por paciente e validação externa"

Ele se divide em 4 repositorio:
**Comum** : Trata-se principalmente dos veirficares e particionares. Essa é uma das partes mais importantes do sistema, é nela que é feita todo tramento de dados garantindo sua conscintencia com os particionamentos, metada dados , classificação. Desse modo podemos garantir que todos os arquivos de teste BRUS-BRA  e  BrEaST-Lesions-USG, não contém nenhum desses erros e que caso algum erro ocorra
posteriomente está em alguma fase que não essa, a sua importância se da por serem a base da metologia desse sistema, garantindo integridade , metodologia e verificação.

**Nota de importância** : Não altere os requirements.txt, já foi testado e comprovado que ao usar bibliotecas de versões diferentes , mesmo rodando com parametros identicos acaba criando uma divergência nos resultados. Essa divergencia até o momento se demonstrou minima, variações de mais ou menos 0.001 . COntudo para manter a metologia e teste de acurácia, principalmente para analises probabilistcias como a Youden e AROC, pequenos valores como esse são de altissima importância.

**Protocolo_AB** : Onde fica o experimento principal, o Protocolo A é o resultado do TCC, validação cruzada com os 5 folds oficiais do BUS-BRA agrupados por paciente. O Protocolo B usa os mesmos dados mas divide por imagem ignorando o paciente de proposito, só pra medir o quanto o vazamento infla o resultado. Aqui tambem ficam o classificador e as metricas que todos os outros reaproveitam.

**Protocolo_C** : Validação externa. Treina uma vez no BUS-BRA inteiro e aplica o modelo congelado no BrEaST-Lesions-USG, que ele nunca viu, inclusive os limiares. Mostra o quanto o resultado cai ao trocar de hospital/aparelho. Tudo dele fica separado em `breast/`, nunca escreve nada do BUS-BRA.

**Resnet** : Caminho de comparação, troca só a extração de caracteristicas por uma ResNet18 pré-treinada (congelada, sem treino) e roda os mesmos Protocolos A e B. Não substitui o classico, grava tudo com `_resnet` no nome.

Cada pasta tem seu proprio README com as funções explicadas.

## Como rodar (Makefile)

make all            # pipeline: indexar -> particionar -> verificar -> features -> experimento -> metricas -> relatorio
make protocolo_c    # BrEaST, só depois do make all
make all_resnet     # caminho da ResNet18 (comparação)
make clean          # apaga dados_processados/ e resultados/
