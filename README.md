# Modelo-probabilisco-de-Analise-de-ultrasson

Este trabalho refe-se ao TCC "Classificação de lesões mamárias em ultrassonografia:
avaliação sob particionamento por paciente e validação externa"

Ele se divide em 4 repositorio:
**Comum** : Trata-se principalmente dos veirficares e particionares. Essa é uma das partes mais importantes do sistema, é nela que é feita todo tramento de dados garantindo sua conscintencia com os particionamentos, metada dados , classificação. Desse modo podemos garantir que todos os arquivos de teste BRUS-BRA  e  BrEaST-Lesions-USG, não contém nenhum desses erros e que caso algum erro ocorra
posteriomente está em alguma fase que não essa, a sua importância se da por serem a base da metologia desse sistema, garantindo integridade , metodologia e verificação.

**Nota de importância** : Não altere os requirements.txt, já foi testado e comprovado que ao usar bibliotecas de versões diferentes , mesmo rodando com parametros identicos acaba criando uma divergência nos resultados. Essa divergencia até o momento se demonstrou minima, variações de mais ou menos 0.001 . COntudo para manter a metologia e teste de acurácia, principalmente para analises probabilistcias como a Youden e AROC, pequenos valores como esse são de altissima importância.
