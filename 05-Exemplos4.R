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
library(shiny)          # intercace gráfica

library(e1071)          # modelo svm
library(glmnet)         # carrega algoritimo de ML
library(xgboost)        # carrega algoritimo de ML

library(h2o)            # framework para construir modelos de machine learning


## Contexto:

# - Imagine que você trabalha em um laboratório de biologia marinha e é responsável por otimizar o processo de determinação da idade de abalones.
#   Atualmente, a idade desses moluscos é determinada através de um método demorado e entediante, que envolve cortar a concha, tingi-la e contar 
#   os anéis microscopicamente. Esse processo consome muito tempo e recursos, tornando-se impraticável para grandes volumes de amostras.

## Problema de Negócio:

# - O desafio é desenvolver um modelo preditivo utilizando técnicas de aprendizado de máquina, especificamente utilizando a linguagem R, para 
#   prever a idade dos abalones com base em medidas físicas mais facilmente obtidas. Dessa forma, o objetivo é criar um sistema mais eficiente 
#   e menos oneroso em comparação ao método tradicional.

# - Para isso, você tem à disposição um conjunto de dados tabular contendo informações sobre 4177 abalones. Esses dados incluem medidas físicas 
#   como diâmetro, altura, peso, entre outros, que foram coletadas de forma mais acessível do que o método de contar anéis. Além disso, o 
#   conjunto de dados já passou por algum pré-processamento, incluindo a remoção de exemplos com valores ausentes e a escala das variáveis
#   contínuas para facilitar o uso em uma Rede Neural Artificial (RNA).

# - A solução desse problema pode impactar positivamente a eficiência do laboratório, permitindo uma estimativa mais rápida e precisa da idade dos
#   abalones sem a necessidade do método tradicional tedioso. Além disso, a inclusão de informações adicionais, como padrões climáticos e
#   localização, pode ser explorada para melhorar ainda mais a precisão do modelo.



## Carregando o dataset

dados <- data.frame(read_csv("datasets/abalone/abalone.data", show_col_types = FALSE))


# Renomeando as colunas
colnames(dados) <- c("Sex", "Length", "Diameter", "Height", "Whole_weight", 
                     "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings")
head(dados)

# Removendo valores ausentes (caso necessário)
dados <- na.omit(dados) 

# Modificando variáveis para tipo factor
dados <- dados %>%
  mutate_if(is.character, factor)

str(dados)
summary(dados)



## SELEÇÃO DE VARIÁVEIS

modelo <- randomForest(Sex ~ ., data = dados, ntree = 100, nodesize = 10, importance = TRUE)

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



## Criando Modelos Preditivos

# Dividindo os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados$Sex, p = 0.80, list = FALSE)  
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)


# Modelo SVM
modelo_svm <- svm(Sex ~ ., data = dados_treino)

# Realizando as previsões
pred_svm <- predict(modelo_svm, newdata = dados_teste)



# Modelo RandomForest
modelo_rf <- randomForest(Sex ~ ., data = dados_treino, ntree = 100, nodesize = 10)
modelo_rf2 <- train(
  Sex ~ .,
  data = dados_treino,
  method = "rf",  # Random Forest
  trControl = trainControl(method = "cv", number = 5)
)
# Realizando as previsões
pred_rf <- predict(modelo_rf, newdata = dados_teste)
pred_rf2 <- predict(modelo_rf2, newdata = dados_teste)



# Modelo Gradient Boosting
modelo_gb <- train(
  Sex ~ .,
  data = dados_treino,
  method = "gbm",  # Gradient Boosting Machine
  trControl = trainControl(method = "cv", number = 5)
)

# Realizando as previsões
pred_gb <- predict(modelo_gb, newdata = dados_teste)



# Modelo k-Nearest Neighbors (k-NN)
modelo_knn <- train(
  Sex ~ .,
  data = dados_treino,
  method = "knn",  # k-Nearest Neighbors
  trControl = trainControl(method = "cv", number = 5)
)

# Realizando as previsões
pred_knn <- predict(modelo_knn, newdata = dados_teste)



# Modelo XGBoost

# Criar a matriz DMatrix para o XGBoost
dados_treino_xg <- dados_treino
dados_teste_xg <- dados_teste

# Codificar as classes numericamente
dados_treino_xg$Sex_numeric <- as.integer(factor(dados_treino$Sex, levels = c("F", "I", "M")))
dados_teste_xg$Sex_numeric <- as.integer(factor(dados_teste$Sex, levels = c("F", "I", "M")))

# Dividir os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados$Sex, p = 0.80, list = FALSE)  
dados_treino_xg <- dados_treino_xg[indices, ]
dados_teste_xg <- dados_teste_xg[-indices, ]
rm(indices)

# Verificar se há valores NA ou NaN na variável de resposta
sum(is.na(dados_treino_xg$Sex_numeric) | is.nan(dados_treino_xg$Sex_numeric))
sum(!is.finite(dados_treino_xg$Sex_numeric))

dados_treino_xg <- dados_treino_xg[complete.cases(dados_treino_xg$Sex_numeric), ]
dados_teste_xg <- dados_teste_xg[complete.cases(dados_teste_xg$Sex_numeric), ]

# Verificar se há valores NA ou NaN na variável de resposta
sum(is.na(dados_treino_xg$Sex_numeric) | is.nan(dados_treino_xg$Sex_numeric))
sum(!is.finite(dados_treino_xg$Sex_numeric))


# Criar a matriz DMatrix para o XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(dados_treino_xg[, -c(1, 9)]), label = dados_treino_xg$Sex_numeric - 1)
dtest <- xgb.DMatrix(data = as.matrix(dados_teste_xg[, -c(1, 9)]), label = dados_teste_xg$Sex_numeric - 1)


# Definir os parâmetros do modelo
parametros <- list(
  objective = "multi:softmax",  # Para problemas de classificação multiclasse
  num_class = 3,                # Número de classes
  eval_metric = "mlogloss"      # Métrica de avaliação
)

# Treinar o modelo XGBoost
modelo_xgb <- xgboost(data = dtrain, params = parametros, nrounds = 100, verbose = 0)


# Realizar previsões no conjunto de teste
pred_xgb <- predict(modelo_xgb, newdata = dtest)




## Avaliando o desempenho dos modelos
confusionMatrix(pred_svm, dados_teste$Sex)   # modelo_svm   -> Accuracy : 0.5719
confusionMatrix(pred_rf, dados_teste$Sex)    # modelo_rf    -> Accuracy : 0.5731 
confusionMatrix(pred_rf2, dados_teste$Sex)   # modelo_rf2   -> Accuracy : 0.5719
confusionMatrix(pred_knn, dados_teste$Sex)   # modelo_knn   -> Accuracy : 0.5336
confusionMatrix(table(pred_xgb, dados_teste_xg$Sex_numeric - 1)) # Accuracy : 1 






## Utilizando AutoML

## Inicializando o H2O (Framework de Machine Learning)

#  -> H20 é um framework de Machine Learning distribuído, no seu ambiente R.
#     O H2O é executado em uma máquina virtual Java (JVM) e fornece uma interface amigável para treinar modelos de Machine Learning.
h2o.init()


# O H2O requer que os dados estejam no formato de dataframe do H2O
h2o_frame <- as.h2o(dados)
class(h2o_frame)
head(h2o_frame)


# Split dos dados em treino e teste (cria duas listas)
h2o_frame_split <- h2o.splitFrame(h2o_frame, ratios = 0.77)
head(h2o_frame_split)


# Modelo AutoML
modelo_automl <- h2o.automl(y = 'Sex',
                            balance_classes = TRUE,
                            training_frame = h2o_frame_split[[1]],
                            nfolds = 4,
                            leaderboard_frame = h2o_frame_split[[2]],
                            max_runtime_secs = 60 * 15,
                            sort_metric = "logloss",               # Use logloss para classificação multiclasse
                            exclude_algos = c("StackedEnsemble"))  # Excluir StackedEnsemble para simplificar


# Extrai o leaderboard (dataframe com os modelos criados)
leaderboard_automl <- as.data.frame(modelo_automl@leaderboard)
h2o.saveModel(modelo_automl@leaderboard$model_id[9], path = "modelos/automl_exemplo4_gbm")


# # Salvar o modelo na 9ª posição do leaderboard
# modelo_id <- leaderboard_automl$model_id[12]
# modelo_salvo <- h2o.getModel(modelo_id)
# h2o.saveModel(modelo_salvo, path = "modelos/automl_exemplo4_glm")
# rm(modelo_id)
# rm(modelo_salvo)


# Carregar modelos
modelo_xgb_1 <- h2o.loadModel(paste0("modelos/automl_exemplo4_lider", "/XGBoost_grid_1_AutoML_1_20240129_201835_model_12"))
modelo_gbm_9 <- h2o.loadModel(paste0("modelos/automl_exemplo4_gbm", "/GBM_grid_1_AutoML_1_20240129_201835_model_2"))
modelo_glm_12 <- h2o.loadModel(paste0("modelos/automl_exemplo4_glm", "/GLM_1_AutoML_1_20240129_201835"))


head(leaderboard_automl, 12)


# Extrai o líder (modelo com melhor performance)
lider_automl <- modelo_automl@leader
modelo_gbm <-leaderboard_automl$model_id[9]
modelo_gblm <- leaderboard_automl$model_id[12]


# Avaliar o desempenho dos modelos
performance <- h2o.performance(modelo_xgb_1, newdata = h2o_frame_split[[2]])
performance_gbm <- h2o.performance(modelo_gbm_9, newdata = h2o_frame_split[[2]])
performance_glm <- h2o.performance(modelo_glm_12, newdata = h2o_frame_split[[2]])

modelo_gbm_9
performance
performance_gbm
performance_glm

# Visualizar a matriz de confusão
h2o.confusionMatrix(performance)



## Desliga o H2O
h2o.shutdown()




## Criando o modelo Foram do Ambiente

library(gbm)

# Converter 'Sex' para fatores numéricos
dados$Sex_numeric <- as.integer(factor(dados$Sex, levels = c("F", "I", "M")))
str(dados)

# Definir as colunas preditoras e a variável de resposta
predictors <- names(dados)[-c(1, 9)]  # Todas as colunas, exceto 'Sex' e 'Sex_numeric'
response <- "Sex_numeric"

# Dividindo os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados$Sex, p = 0.80, list = FALSE)  
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)

# Configurar os parâmetros do modelo GBM
gbm_params <- list(
  distribution = "multinomial",
  n.trees = 37,
  interaction.depth = 9,
  shrinkage = 0.2,
  n.minobsinnode = 10
)

# Treinar o modelo GBM
modelo_gbm_manual <- gbm(
  formula = as.formula(paste(response, "~", paste(predictors, collapse = " + "))),
  data = dados_treino,
  distribution = gbm_params$distribution,
  n.trees = gbm_params$n.trees,
  interaction.depth = gbm_params$interaction.depth,
  shrinkage = gbm_params$shrinkage,
  n.minobsinnode = gbm_params$n.minobsinnode,
  cv.folds = 4,
  verbose = TRUE
)

# Exibir o resumo do modelo
summary(modelo_gbm_manual)


# Fazer previsões no conjunto de teste
predicoes_teste <- predict(modelo_gbm_manual, dados_teste, n.trees = 37)

# MSE (Erro Quadrático Médio)
mse_teste <- mean((dados_teste$Sex_numeric - predicoes_teste)^2)
cat("MSE Teste:", mse_teste, "\n")

# RMSE (Raiz do Erro Quadrático Médio)
rmse_teste <- sqrt(mse_teste)
cat("RMSE Teste:", rmse_teste, "\n")

# Logloss
# Como mencionado anteriormente, não há uma função direta para calcular logloss no pacote gbm. Você pode usar pacotes adicionais ou implementar manualmente.

# Mean Per-Class Error
# Converter as previsões em classes preditas (usando regras específicas para problemas multinomiais)
classes_preditas_teste <- colnames(predicoes_teste)[apply(predicoes_teste, 1, which.max)]
mean_per_class_error_teste <- mean(classes_preditas_teste != dados_teste$Sex_numeric)
cat("Mean Per-Class Error Teste:", mean_per_class_error_teste, "\n")



class(modelo_gbm_9)
class(modelo_gbm_manual)



## Comparando Modelos

# Para modelo_gbm_9
h2o.init()

# Para modelo_gbm_9
performance_gbm_9 <- h2o.performance(modelo_gbm_9, newdata = h2o_frame_split[[2]])

# Para modelo_gbm_manual
predicoes_teste_manual <- predict(modelo_gbm_manual, dados_teste, n.trees = 37)
mse_teste_manual <- mean((dados_teste$Sex_numeric - predicoes_teste_manual)^2)
rmse_teste_manual <- sqrt(mse_teste_manual)

# Outras métricas (se disponíveis ou implementadas)
# ...

# Comparação
cat("Modelo gbm_9:\n")
print(performance_gbm_9)

cat("\nModelo gbm_manual:\n")
cat("MSE Teste:", mse_teste_manual, "\n")
cat("RMSE Teste:", rmse_teste_manual, "\n")
# ...




