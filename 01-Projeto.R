####  Objetivo e Definição do Problema de Negócio  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/19.Mini-Projeto-3_-_Explicabilidade_de_Modelos_AutoML_com_SHAP")
getwd()


## Carregando pacotes
library(h2o)
library(tidyverse)
library(ggbeeswarm)


## Criando Dados (como os dados serão gerados de forma randômica, o resultado será diferente a cada execução)
dados <- tibble(produtividade = c(rnorm(1000), rnorm(1000, 0.25)),
                rendimento = runif(2000),
                custo = rf(2000, df1 = 5, df2 = 2),
                prioridade = c(sample(rep(c('Baixa', 'Media', 'Alta'), c(300, 300, 400))), 
                               sample(c('Baixa', 'Media', 'Alta'), 1000, prob = c(0.25, 0.25, 0.5), replace = T)),
                eficiencia = rnorm(2000),
                manutencao = rep(c(0,1), c(1050,950)))

# - 0 significa que o equipamento não requer manutenção (não)
# - 1 significa que o equipamento requer manutenção (sim)

