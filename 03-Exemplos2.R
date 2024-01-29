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



## Declaração do Problema: Previsão da Intensidade de Erupções Solares

## Contexto:

# - As erupções solares são explosões repentinas e intensas de energia na superfície do Sol que podem impactar a meteorologia espacial e ter
#   efeitos potenciais nos sistemas tecnológicos da Terra. Cientistas frequentemente estudam a atividade de erupções solares para entender 
#   melhor esses fenômenos e prever sua intensidade. Nesse contexto, temos um conjunto de dados que captura várias características de regiões 
#   ativas no Sol e tem como objetivo prever a intensidade das erupções solares nas próximas 24 horas.

## Objetivo:

# - Construir um modelo preditivo usando o conjunto de dados fornecido para classificar as regiões ativas no Sol em três classes com base na
#   intensidade das erupções solares que provavelmente produzirão nas próximas 24 horas: C-class, M-class e X-class.

## Informações do Conjunto de Dados:

# - O conjunto de dados consiste em 1389 instâncias, cada uma representando características de uma região ativa no Sol. As características incluem
#   variáveis categóricas, como a classe Zurich modificada, tamanho do ponto mais amplo, distribuição de pontos, atividade, evolução e complexidade 
#   histórica, entre outras. As últimas três colunas representam as variáveis-alvo: erupções solares da classe C, erupções solares da classe M e
#   erupções solares da classe X, produzidas nas próximas 24 horas.



## Carregando dataset

dados <- data.frame(read_csv("datasets/solar+flare/flare.data1"))
dados2 <- data.frame(read_csv("datasets/solar+flare/flare.data2"))

head(dados)
head(dados2)
colnames(dados)
colnames(dados2)

## Análise Exploratória

# Adicionar uma coluna "data" aos dois conjuntos de dados
dados$data <- "02.13.69.to.03.27.69"
dados2$data <- "08.19.78.to.12.23.78"

# Selecionar apenas as colunas necessárias de cada conjunto de dados
dados_selecionados <- dados %>% select(data, X.......DATA1..1969.FLARE.DATA...02.13.69.to.03.27.69.........)
dados2_selecionados <- dados2 %>% select(data, X........DATA2..1978.FLARE.DATA..08.19.78.to.12.23.78.......)

# Renomear as colunas para simplificar
colnames(dados_selecionados) <- c("data", "dados")
colnames(dados2_selecionados) <- c("data", "dados")

# Unir os datasets
rm(dados)
dados <- bind_rows(dados_selecionados, dados2_selecionados)
rm(dados_selecionados)
rm(dados2_selecionados)
rm(dados2)

table(dados$data)



# Separar a coluna 'dados' em colunas individuais
dados <- dados %>%
  mutate(
    cod_classe = substr(dados, 1, 1),
    cod_tam_mancha_solar = substr(dados, 3, 3),
    cod_dist_manchas_solares = substr(dados, 5, 5),
    atividade = substr(dados, 7, 7),
    evolucao = substr(dados, 9, 9),
    cod_atividade_flare_24hs = substr(dados, 11, 11),
    historicamente_complexo = substr(dados, 13, 13),
    regiao_complexa_disco_solar = substr(dados, 15, 15),
    area = substr(dados, 17, 17),
    area_maior_mancha = substr(dados, 19, 19)
  ) %>%
  select(data, cod_classe, cod_tam_mancha_solar, cod_dist_manchas_solares,
         atividade, evolucao, cod_atividade_flare_24hs,
         historicamente_complexo, regiao_complexa_disco_solar,
         area, area_maior_mancha)
  

# Modificando variáveis para tipo factor
dados <- dados %>%
  mutate_if(is.character, factor)

# Visualizar as primeiras linhas dos dados transformados
head(dados)

# Tipo de Dados
dim(dados)
str(dados)
summary(dados)

table(dados$cod_classe)



## Um modelo que preveja o codigo de classe com base nas informações das variáveis preditoras. Um modelo para prever o número de erupções 
#  solares da classe C. Um modelo que  Representa o número de erupções solares da classe D e um outro modelo que Representa o número de
#  erupções solares da classe H.


# Verificando valores ausentes
dados <- na.omit(dados)        # Remove linhas com valores ausentes



# Dividir o conjunto de dados em treino e teste (para ser usado na primeira versão do modelo sem variáveis específicas para o cod de classe)
set.seed(123)
split_index <- createDataPartition(dados$cod_classe, p = 0.80, list = FALSE)
train_data_original <- dados[split_index, ]
test_data_original <- dados[-split_index, ]
rm(split_index)



# Adicionar novas variáveis
#dados$cod_c <- ifelse(dados$cod_classe == "C", "sim", "não")
#dados$cod_d <- ifelse(dados$cod_classe == "D", "sim", "não")
#dados$cod_h <- ifelse(dados$cod_classe == "H", "sim", "não")

# Modificando variáveis para tipo factor
dados <- dados %>%
  mutate_if(is.character, factor)

str(dados)
summary(dados)


## Criando Modelo Para Prever o Código de Classe
dados_mod1 <- dados
names(dados_mod1)

dados_mod1 <- dados %>% 
  select(cod_classe, cod_tam_mancha_solar, cod_dist_manchas_solares, atividade, evolucao, historicamente_complexo,
         regiao_complexa_disco_solar, area, area_maior_mancha)


# Seleção de Variáveis
modelo <- randomForest(cod_classe ~ ., 
                       data = dados_mod1, 
                       ntree = 100, nodesize = 10, importance = T)

# Visualizando por números
print(modelo$importance)

# Visualizando Modelo Por Gráficos

# forma 1 (quanto mais a direita melhor)
varImpPlot(modelo)

# forma 2 (quando tem poucas variáveis)
barplot(modelo$importance[, 1], main = "Importância das Variáveis", col = "skyblue")      

# forma 3 (usando ggplot, método mais profissional)
importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 

df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))

str(dados_mod1)

dados_mod1 <- dados %>% 
  select(cod_classe, cod_tam_mancha_solar, cod_dist_manchas_solares, historicamente_complexo,
         regiao_complexa_disco_solar, area, area_maior_mancha)


# Dividir o conjunto de dados em treino e teste
set.seed(123)
split_index <- createDataPartition(dados_mod1$cod_classe, p = 0.80, list = FALSE)
train_data <- dados_mod1[split_index, ]
test_data <- dados_mod1[-split_index, ]
rm(split_index)

table(train_data$cod_classe)
table(test_data$cod_classe)


# Modelo para prever o código de classe
model_classe <- train(
  cod_classe ~ .,
  data = train_data_original,
  method = "rf",  # Random Forest
  trControl = trainControl(method = "cv", number = 5)
)

model_classe_v2 <- train(
  cod_classe ~ .,
  data = train_data,
  method = "rf",  # Random Forest
  trControl = trainControl(method = "cv", number = 5)
)




## Criando Modelo para prever o número de erupções solares da classe C
names(dados)
dados_mod2 <- dados %>% 
  select(cod_classe, cod_tam_mancha_solar, cod_dist_manchas_solares, atividade, evolucao, historicamente_complexo,
         regiao_complexa_disco_solar, area, area_maior_mancha)

dados_mod2$cod_c <- ifelse(dados_mod2$cod_classe == "C", "sim", "não")

dados_mod2 <- dados_mod2 %>% 
  select(-cod_classe) %>% 
  mutate_if(is.character, factor)



# Dividir o conjunto de dados em treino e teste
set.seed(123)
split_index <- createDataPartition(dados_mod2$cod_c, p = 0.80, list = FALSE)
train_data <- dados_mod2[split_index, ]
test_data <- dados_mod2[-split_index, ]
rm(split_index)


# Modelo para prever o código de classe C
model_C <- train(
  atividade ~ .,
  data = filter(train_data_original, cod_classe == "C"),
  method = "glmnet",  # Elastic Net
  trControl = trainControl(method = "cv", number = 5)
)

model_C_v2 <- train(
  cod_c ~ .,
  data = train_data,
  method = "glmnet",  # Elastic Net
  trControl = trainControl(method = "cv", number = 5)
)



## Criando Modelo para prever o número de erupções solares da classe D
names(dados)
dados_mod3 <- dados %>% 
  select(cod_classe, cod_tam_mancha_solar, cod_dist_manchas_solares, atividade, evolucao, historicamente_complexo,
         regiao_complexa_disco_solar, area, area_maior_mancha)

dados_mod3$cod_d <- ifelse(dados_mod3$cod_classe == "D", "sim", "não")

dados_mod3 <- dados_mod3 %>% 
  select(-cod_classe) %>% 
  mutate_if(is.character, factor)



# Dividir o conjunto de dados em treino e teste
set.seed(123)
split_index <- createDataPartition(dados_mod3$cod_d, p = 0.80, list = FALSE)
train_data <- dados_mod3[split_index, ]
test_data <- dados_mod3[-split_index, ]
rm(split_index)


# Modelo para prever o código de classe D
model_D <- train(
  atividade ~ .,
  data = filter(train_data_original, cod_classe == "D"),
  method = "glmnet",  # Elastic Net
  trControl = trainControl(method = "cv", number = 5)
)

model_D_v2 <- train(
  cod_d ~ .,
  data = train_data,
  method = "glmnet",  # Elastic Net
  trControl = trainControl(method = "cv", number = 5)
)



## Criando Modelo para prever o número de erupções solares da classe H
names(dados)
dados_mod4 <- dados %>% 
  select(cod_classe, cod_tam_mancha_solar, cod_dist_manchas_solares, atividade, evolucao, historicamente_complexo,
         regiao_complexa_disco_solar, area, area_maior_mancha)

dados_mod4$cod_h <- ifelse(dados_mod4$cod_classe == "H", "sim", "não")

dados_mod4 <- dados_mod4 %>% 
  select(-cod_classe) %>% 
  mutate_if(is.character, factor)



# Dividir o conjunto de dados em treino e teste
set.seed(123)
split_index <- createDataPartition(dados_mod4$cod_h, p = 0.80, list = FALSE)
train_data <- dados_mod4[split_index, ]
test_data <- dados_mod4[-split_index, ]
rm(split_index)


# Modelo para prever o código de classe H
model_H <- train(
  atividade ~ .,
  data = filter(train_data_original, cod_classe == "H"),
  method = "glmnet",  # Elastic Net
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(alpha = seq(0, 1, by = 0.1), lambda = seq(0.001, 1, by = 0.01)),
  preProcess = c("center", "scale"),
  tuneLength = 10,  # Número de valores lambda a serem testados
  na.action = na.omit  # Remover NAs nas variáveis preditoras
)

model_H_v2 <- train(
  cod_h ~ .,
  data = train_data,
  method = "glmnet",  # Elastic Net
  trControl = trainControl(method = "cv", number = 5)
)





# Avaliar os modelos nos dados de teste
predictions_classe <- predict(model_classe, newdata = test_data_original)
predictions_classe_v2 <- predict(model_classe_v2, newdata = test_data)

predictions_C <- predict(model_C, newdata = test_data_original)
predictions_C_C <- predictions_C[test_data_original$cod_classe == "C"]
predictions_C_v2 <- predict(model_C_v2, newdata = test_data)

predictions_D <- predict(model_D, newdata = test_data_original)
predictions_D_D <- predictions_D[test_data_original$cod_classe == "D"]
predictions_D_v2 <- predict(model_D_v2, newdata = test_data)

predictions_H <- predict(model_H, newdata = test_data_original)
predictions_H_H <- predictions_H[test_data_original$cod_classe == "H"]
predictions_H_v2 <- predict(model_H_v2, newdata = test_data)


# Avaliação da acurácia do modelo de classe
confusionMatrix(predictions_classe, test_data_original$cod_classe)  # 94%
confusionMatrix(predictions_classe_v2, test_data$cod_classe)        # 75%

# Avaliação da acurácia do modelo de classe C
confusionMatrix(predictions_C_C, subset(test_data_original, cod_classe == "C")$atividade)     # 98%
confusionMatrix(predictions_C_v2, test_data$cod_c)                                            # 83%

# Avaliação da acurácia do modelo de classe D
confusionMatrix(predictions_D_D, filter(test_data_original, cod_classe == "D")$atividade)     # 86%
confusionMatrix(predictions_D_v2, test_data$cod_d)                                            # 80%

# Avaliação da acurácia do modelo de classe H
confusionMatrix(predictions_H_H, filter(test_data_original, cod_classe == "H")$atividade)     # 98%
confusionMatrix(predictions_H_v2, test_data$cod_h)                                            # 100%






