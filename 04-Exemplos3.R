####  MAIS_EXEMPLOS  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/19.Mini-Projeto-3_-_Explicabilidade_de_Modelos_AutoML_com_SHAP")
getwd()


## Carregar pacotes

library(tidyverse)      # manipulação de dados
library(ggbeeswarm)     # pacote que permite criar gráficos customizados sobre o ggplot2
library(ggplot2)        # gera gráficos
library(dplyr)          # manipulação de dados
library(randomForest)   # carrega algoritimo de ML
library(ROCR)           # Gerando uma curva ROC em R
library(caret)          # cria confusion matrix
library(readxl)         # carregar arquivos 
library(glmnet)         # carrega algoritimo de ML
library(corrplot)       # criar gráfico de mapa de correlação
library(shiny)          # intercace gráfica
library(lubridate)


## Contexto:

# - O conjunto de dados em questão consiste em informações sobre o volume de tráfego na rodovia Interstate 94 Westbound em Minneapolis-St Paul,
#   MN, coletadas de forma horária. O período abrange os anos de 2012 a 2018 e inclui características relacionadas ao clima e feriados.
#   A estação de monitoramento, MN DoT ATR 301, está localizada aproximadamente no meio do trajeto entre Minneapolis e St Paul, MN.

# - Este é um conjunto de dados multivariado, sequencial e de séries temporais, contendo 48.204 instâncias. As informações fornecem insights não 
#   apenas sobre o tráfego, mas também sobre as condições meteorológicas e feriados que podem influenciar o volume de tráfego na rodovia.

## Problema de Negócio:
  
# - O problema de negócio aqui é desenvolver um modelo de regressão capaz de prever o volume horário de tráfego na Interstate 94 Westbound.
#   Isso implica considerar as variáveis relacionadas ao clima, como temperatura, quantidade de chuva e neve, percentual de cobertura de nuvens,
#   bem como informações sobre feriados.

# - Uma previsão precisa do volume de tráfego pode ser valiosa para diversas aplicações, como o gerenciamento do tráfego rodoviário, o 
#   planejamento de operações de manutenção e a otimização de recursos em situações de tráfego intenso. Além disso, compreender como as condições
#   meteorológicas e feriados afetam o tráfego pode ser útil para tomadas de decisões estratégicas em planejamento urbano e gestão de transporte. 

# - O objetivo final é criar um modelo que forneça previsões confiáveis do volume de tráfego com base nas condições horárias específicas do 
#   ambiente.



## Carregando o dataset

dados <- data.frame(read_csv("datasets/metro/metro.csv", show_col_types = FALSE))



## Análise Exploratória

# Renomeando as colunas
nomes_colunas <- c("feriado", "temperatura_media", "chuva_1h", "neve_1h", "cobertura_nuvens", 
                   "condicao_climatica_principal", "descricao_climatica", "data_hora", "volume_trafego")

# Aplicando os novos nomes às colunas
dados <- rename(dados, !!!setNames(names(dados), nomes_colunas))
rm(nomes_colunas)

# Removendo valores ausentes (caso necessário)
dados <- na.omit(dados) 

# Modificando variáveis para tipo factor
dados <- dados %>%
   mutate_if(is.character, factor)

dados_interface <- dados
str(dados)
summary(dados)

# Tipo de dados
str(dados)
summary(dados)
dim(dados)

levels(dados$descricao_climatica)

## Tratamento de Variáveis Categóricas (feriado, condicao_climatica_principal, descricao_climatica)

# Função de codificação de contagem
count_encode <- function(x) {
  as.integer(table(x)[match(x, names(table(x)))])
}

# Converter variáveis categóricas usando codificação de contagem
dados$feriado <- count_encode(dados$feriado)
dados$condicao_climatica_principal <- count_encode(dados$condicao_climatica_principal)
dados$descricao_climatica <- count_encode(dados$descricao_climatica)
rm(count_encode)

# Tipo de dados
str(dados)
summary(dados)
head(dados)


## Tratamento da variável data_hora

# Extrair informações relevantes da variável data_hora
dados <- dados %>%
  mutate(ano = lubridate::year(data_hora),
         mes = lubridate::month(data_hora),
         dia_semana = lubridate::wday(data_hora),
         hora_dia = lubridate::hour(data_hora))

# Excluir a variável data_hora original (opcional, dependendo da preferência)
dados <- select(dados, -data_hora)
head(dados)


## Normalização das variáveis numéricas

# Selecionar as variáveis numéricas
numeric_vars <- c("temperatura_media", "chuva_1h", "neve_1h", "cobertura_nuvens", "volume_trafego", "ano", "mes", "dia_semana", "hora_dia")

# Normalizar as variáveis numéricas
dados[numeric_vars] <- scale(dados[numeric_vars])
rm(numeric_vars)

head(dados)
str(dados)


## SELEÇÃO DE VARIÁVEIS

modelo <- randomForest(volume_trafego ~ ., data = dados, ntree = 100, nodesize = 10, importance = TRUE)

print(modelo$importance)
varImpPlot(modelo)
importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 
df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))
rm(df_importancia)
rm(importancia_ordenada)
rm(modelo)

dados_sel_var <- dados %>% 
  select(-chuva_1h, -feriado, -neve_1h)



## Criando Modelos Preditivos

# Dividindo os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados$volume_trafego, p = 0.80, list = FALSE)  
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)
indices <- createDataPartition(dados_sel_var$volume_trafego, p = 0.80, list = FALSE)  
dados_treino_sel_var <- dados_sel_var[indices, ]
dados_teste_sel_var <- dados_sel_var[-indices, ]
rm(indices)


# Modelo (todas as variáveis)
model_rand <- randomForest(volume_trafego ~ .,
                          data = dados_treino,
                          ntree = 40, 
                          nodesize = 5)
model_rand

model_lm <- lm(volume_trafego ~ ., data = dados_treino)
summary(model_lm)


# Modelo (seleção de variáveis)
model_rand_sel_var <- randomForest(volume_trafego ~ .,
                           data = dados_treino_sel_var,
                           ntree = 40, 
                           nodesize = 5)
model_rand

model_lm_sel_var <- lm(volume_trafego ~ ., data = dados_treino_sel_var)
summary(model_lm)




## Avaliação dos Modelos

# Para o modelo de Random Forest (todas as variáveis)
predicted_rf <- predict(model_rand, newdata = dados_teste)
rmse_rf <- sqrt(mean((predicted_rf - dados_teste$volume_trafego)^2))
mae_rf <- mean(abs(predicted_rf - dados_teste$volume_trafego))
r_squared_rf <- 1 - sum((dados_teste$volume_trafego - predicted_rf)^2) / sum((dados_teste$volume_trafego - mean(dados_teste$volume_trafego))^2)

# Para o modelo de Regressão Linear (todas as variáveis)
predicted_lm <- predict(model_lm, newdata = dados_teste)
rmse_lm <- sqrt(mean((predicted_lm - dados_teste$volume_trafego)^2))
mae_lm <- mean(abs(predicted_lm - dados_teste$volume_trafego))
r_squared_lm <- summary(model_lm)$r.squared

# Exibindo os resultados
cat("Random Forest - RMSE:", rmse_rf, "\n")                  # 0.2357508
cat("Random Forest - MAE:", mae_rf, "\n")                    # 0.1551042 
cat("Random Forest: - R-squared:", r_squared_rf, "\n\n")     # 0.9442402

cat("Regressão Linear - RMSE:", rmse_lm, "\n")               # 0.9185076
cat("Regressão Linear - MAE:", mae_lm, "\n")                 # 0.8123023
cat("Regressão Linear - R-squared:", r_squared_lm, "\n")     # 0.1473663


# Para o modelo de Random Forest (seleção de variáveis)
predicted_rf <- predict(model_rand_sel_var, newdata = dados_teste_sel_var)
rmse_rf <- sqrt(mean((predicted_rf - dados_teste_sel_var$volume_trafego)^2))
mae_rf <- mean(abs(predicted_rf - dados_teste_sel_var$volume_trafego))
r_squared_rf <- 1 - sum((dados_teste_sel_var$volume_trafego - predicted_rf)^2) / sum((dados_teste_sel_var$volume_trafego - mean(dados_teste_sel_var$volume_trafego))^2)

# Para o modelo de Regressão Linear (seleção de variáveis)
predicted_lm <- predict(model_lm_sel_var, newdata = dados_teste_sel_var)
rmse_lm <- sqrt(mean((predicted_lm - dados_teste_sel_var$volume_trafego)^2))
mae_lm <- mean(abs(predicted_lm - dados_teste_sel_var$volume_trafego))
r_squared_lm <- summary(model_lm_sel_var)$r.squared

# Exibindo os resultados
cat("Random Forest - RMSE:", rmse_rf, "\n")                  # 0.2420281
cat("Random Forest - MAE:", mae_rf, "\n")                    # 0.1629034 
cat("Random Forest: - R-squared:", r_squared_rf, "\n\n")     # 0.9419506

cat("Regressão Linear - RMSE:", rmse_lm, "\n")               # 0.9274825
cat("Regressão Linear - MAE:", mae_lm, "\n")                 # 0.8195713
cat("Regressão Linear - R-squared:", r_squared_lm, "\n")     # 0.1483796

rm(predicted_rf)
rm(rmse_rf)
rm(mae_rf)
rm(r_squared_rf)
rm(predicted_lm)
rm(rmse_lm)
rm(mae_lm)
rm(r_squared_lm)



saveRDS(model_rand, file = "modelos/modelo_randomForest_ex3.rds")



