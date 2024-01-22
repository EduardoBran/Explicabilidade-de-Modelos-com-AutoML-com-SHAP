####  Objetivo e Definição do Problema de Negócio  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/19.Mini-Projeto-3_-_Explicabilidade_de_Modelos_AutoML_com_SHAP")
getwd()


#### Explicabilidade de Modelos com AutoML com SHAP (SHapley Additive exPlanations)


## Objetivo

# - Não adianta apenas saber construir modelos de Machine Learning. É preciso saber explicar como eles chegam
#   aos seus resultados. Sem isso, dificilmente um gestor sentirá confiança no uso de um modelo preditivo.

# - Neste mini-projeto o objetivo é exatamente demonstrar na prática como explicar o resultado de um modelo de
#   Machine Learning. 
#   E para deixar as coisas mais interessantes o modelo será criado através de AutoML.
#   E a explicabilidade será feita através de valores SHAP.

# - Trabalharemo no contexto de um problema na área de manutenção de máquinas industriais.


## O que é AutoML ?

# - AutoML ou Automated Machine Learning é o processo de automatizar as tarefas do desenvolimento de modelos de
#   Machine Learning. Com AutoML, Cientistas de Dados podem criar modelos de ML com alta escala, eficiência e
#   produtividade, ao mesmo tempo em dão suporte à qualidade do modelo.

# - O desenvolvimento do modelo de Machine Learning tradicional tem uso intensivo de recursos, exigindo conhecimento
#   de domínio significativo e tempo para produzir e comparar dezenas de modelos.

# - Com o AutoML, você vai acelerar o tempo necessário para obter modelos de ML prontos para produção com grande
#   facilidade e eficiência.


## Definição do Problema de Negócio e Coleta de Dados

# - Uma empresa produz itens hospitalares através de uma das suas fábricas no Brasil. Cada fábrica possui diversos
#   equipamentos industriais que periodicamente requerem manutenção.

# - A empresa coletou dados históricos associando diferentes métricas (variáveis preditoras) à necessidade de manutenção
#   do equipamento (sim ou não). A idéia é ter um modelo de Machine Learning capaz de prever quando cada máquina vai
#   requerer manutenção e assim evitar paradas não programadas.

# - Mas antes de usar um modelo preditivo a alta gerência necessita compreender como o modelo faz as previsões e quais
#   métricas tem maior impacto na previsão do modelo.

# - Você foi convidado a fazer uma apresentação sobre o tema! Qual seria o processo para responder às dúvidas da
#   alta gerência?


## Fonte de Dados

# - Para este mini-projeto será criado uma massa de dados fictícios, que repesentam dados reais.
#   Abaixo o dicionário de dados:

# Variável Preditora 1 (produtividade)

# - Eficácia geral do equipamento: Está é uma medida de produtividade, que descreve a parte do tempo em que
#   uma máquina trabalha com desempenho máximo. A métrica é um produto da disponibilidade, desempenho e qualidade da
#   máquina.

# Variável Preditora 2 (rendimento)

# - Rendimento de primeira passagem: Esta é a parcela de produtos que saem da linha de produção e que não apresentam e
#   que não apresentam defeitos e atendem às especificações sem a necessidade de qualquer trabalho de retificação

# Variável Preditora 3 (custo)

# - Custo de energia por unidade: Este é o custo da eletrecidade, vapor, óleo ou gás necessário para produzir uma determinada
#   unidade de produto na fábrica.

# Variável Preditora 4 (prioridade)

# - Prioridade do equipamento quando entrar no período de manutenção (Baixa, Média, Alta).

# Variável Preditora 5 (eficiencia)

# - A quantidade de produto que uma máquina produz durante um período específico. Essa métrica também pode ser
#   aplicada a toda a linha de produção para verificar sua eficiência.

# Variável Alvo (manutenção)

# - 0 significa que o equipamento não requer manutenção (não)
# - 1 significa que o equipamento requer manutenção (sim)


