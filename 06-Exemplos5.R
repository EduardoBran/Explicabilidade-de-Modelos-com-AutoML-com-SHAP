####  MAIS_EXEMPLOS  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/19.Mini-Projeto-3_-_Explicabilidade_de_Modelos_AutoML_com_SHAP")
getwd()


## Carregar pacotes

library(readxl)         # carregar arquivos 
library(dplyr)          # manipulação de dados

library(ggplot2)        # gera gráficos
library(shiny)          # intercace gráfica

library(randomForest)   # carrega algoritimo de ML (randomForest)
library(rpart)          # carrega algoritimo de ML (árvore de decisão)
library(e1071)          # carrega algoritimo de ML (SVM)
library(xgboost)        # carrega algoritimo de ML (Gradient Boosting - XGBoost)

library(ROCR)           # Gerando uma curva ROC em R
library(caret)          # cria confusion matrix
library(vip)            # visualização de importância de variáveis

library(h2o)            # framework para construir modelos de machine learning

## Contexto:

# - Você é um cientista de dados em uma empresa de desenvolvimento de jogos, e a equipe de design de jogos está interessada em entender padrões
#   e estratégias que levam à vitória no jogo da velha. O objetivo é criar um modelo de machine learning capaz de prever a vitória para o
#   jogador "X" com base nas configurações finais do tabuleiro.

## Dados:

# - O conjunto de dados dados contém informações sobre diferentes configurações finais de tabuleiros de jogo da velha. Cada linha representa um
#   estado final de jogo, onde "X" é assumido como o primeiro jogador.
#   As características incluem os conteúdos das nove posições do tabuleiro (top_left_square, top_middle_square, ..., bottom_right_square), e a
#   variável alvo é "Class", indicando se o jogador "X" ganhou (positive) ou não (negative).

## Problema de Negócio:
  
# - A equipe de design está buscando insights sobre as estratégias que levam a vitórias no jogo da velha. Eles desejam incorporar esses insights 
#   no desenvolvimento de um jogo mais desafiador e envolvente. O modelo de machine learning proposto será utilizado para prever se um jogador "X" 
#   vencerá com base na configuração atual do tabuleiro.

# Perguntas de Negócio:
  
# 1) O modelo consegue identificar estratégias eficazes ou movimentos que frequentemente resultam em vitórias?
# 2) Há configurações específicas que indicam uma alta probabilidade de derrota para "X"?
# 3) Quais insights o modelo pode fornecer para melhorar a experiência do jogador e aumentar o desafio do jogo?
  
## Abordagem:

# - Utilizando o conjunto de dados, você explorará diferentes algoritmos de classificação em machine learning para desenvolver um modelo preciso
#   na previsão de vitórias para o jogador "X". Além disso, realizará análises de importância de características para identificar as posições do
#   tabuleiro mais influentes nas vitórias.

# - Ao finalizar o projeto, você apresentará insights à equipe de design, permitindo que eles aprimorem o jogo da velha, incorporando estratégias 
#   autênticas e desafiadoras, oferecendo uma experiência mais interessante aos jogadores.



## Carregando o dataset

dados <- data.frame(read_csv("datasets/tic_tac_toe_endgame/tic-tac-toe.data", show_col_types = FALSE))

str(dados)

# Renomear as variáveis
nomes_variaveis <- c(
  "top_left_square", "top_middle_square", "top_right_square",
  "middle_left_square", "middle_middle_square", "middle_right_square",
  "bottom_left_square", "bottom_middle_square", "bottom_right_square",
  "Class"
)

# Atribuir os nomes corretos às variáveis
colnames(dados) <- nomes_variaveis
rm(nomes_variaveis)

# Tipo de Dados
str(dados)
summary(dados)

# Transformar todas as variáveis em dados_fac para tipo fator
dados_fac <- dados
dados_fac[] <- lapply(dados_fac, as.factor)

# Transformar todas as variáveis em dados_int para tipo inteiro menos Class
dados_int <- dados_fac
dados_int[, -10] <- lapply(dados_int[, -10], as.integer)

# Tipo de Dados
str(dados_fac)
str(dados_int)
summary(dados_fac)
summary(dados_int)

head(dados_fac)


##  Aplicando Balanceamento

# Forma 1 (Função para equilibrar as classes duplicando linhas da classe minoritária)
equilibraClasses <- function(df, target_column) {
  # Identifica a classe majoritária e minoritária
  classes <- levels(df[[target_column]])
  majority_class <- ifelse(sum(df[[target_column]] == classes[1]) > sum(df[[target_column]] == classes[2]), classes[1], classes[2])
  
  # Calcula o número de amostras da classe majoritária
  majority_samples <- sum(df[[target_column]] == majority_class)
  
  # Amostra aleatória da classe minoritária
  minority_rows <- df[df[[target_column]] != majority_class, ]
  minority_rows <- minority_rows[sample(nrow(minority_rows), majority_samples, replace = TRUE), ]
  
  # Junta as linhas da classe minoritária amostradas com o dataframe original
  balanced_df <- rbind(df[df[[target_column]] == majority_class, ], minority_rows)
  
  return(balanced_df)
}

# Aplica a função para equilibrar as classes
dados_fac <- equilibraClasses(dados_fac, "Class")
dados_int <- equilibraClasses(dados_int, "Class")
rm(equilibraClasses)

str(dados_fac)
str(dados_int)
summary(dados_fac)
summary(dados_int)


# Salvar df_fac_bal como CSV
#write.csv(df_fac_bal, file = "datasets/tic_tac_toe_endgame/tic-tac-toe_edit.csv", row.names = FALSE)

# Salvar df_int_bal como CSV
#write.csv(df_int_bal, file = "datasets/tic_tac_toe_endgame/tic-tac-toe_int_edit.csv", row.names = FALSE)


## Carregar Dataset Editado Balanceado
dados_fac <- data.frame(read_csv("datasets/tic_tac_toe_endgame/tic-tac-toe_edit.csv", show_col_types = FALSE))
dados_int <- data.frame(read_csv("datasets/tic_tac_toe_endgame/tic-tac-toe_int_edit.csv", show_col_types = FALSE))

# Transformar todas as variáveis em dados_fac para tipo fator
dados_fac[] <- lapply(dados_fac, as.factor)

# Transformar todas as variáveis em dados_int para tipo inteiro menos Class
dados_int <- dados_fac
dados_int[, -10] <- lapply(dados_int[, -10], as.integer)

# Tipo de Dados
str(dados_fac)
str(dados_int)
summary(dados_fac)
summary(dados_int)




# Dividindo os dados em treino e teste
set.seed(150)
indices_fac <- createDataPartition(dados_fac$Class, p = 0.80, list = FALSE)  
indices_int <- createDataPartition(dados_int$Class, p = 0.80, list = FALSE)  
dados_treino_fac <- dados_fac[indices_fac, ]
dados_treino_int <- dados_int[indices_int, ]
dados_teste_fac<- dados_fac[-indices_fac, ]
dados_teste_int <- dados_int[-indices_int, ]
rm(indices_fac)
rm(indices_int)


#### Criação dos Modelos Pra Prever Variável Class


## Modelo 1: Árvore de Decisão
modelo_arvore <- rpart(Class ~ ., data = dados_treino_fac, method = "class")

# Faça previsões
previsoes_arvore <- predict(modelo_arvore, newdata = dados_teste_fac, type = "class")

# Avalie a acurácia do modelo
acuracia_arvore <- sum(previsoes_arvore == dados_teste_fac$Class) / nrow(dados_teste_fac)
print(paste("Acurácia Árvore de Decisão (Teste): ", acuracia_arvore))                       # "Acurácia Árvore de Decisão (Teste):  0.936"
rm(previsoes_arvore)
rm(acuracia_arvore)



## Modelo 2: Random Forest
modelo_rf <- randomForest(Class ~ ., data = dados_treino_fac)

# Faça previsões
previsoes_rf <- predict(modelo_rf, newdata = dados_teste_fac)

# Calcular a importância das variáveis no modelo RandomForest
importancia_variaveis <- randomForest::importance(modelo_rf)
importancia_variaveis

# Visualizar a importância das variáveis
vip::vip(modelo_rf, num_features = 5)
rm(importancia_variaveis)

# Avalie a acurácia do modelo
acuracia_rf <- sum(previsoes_rf == dados_teste_fac$Class) / nrow(dados_teste_fac)
print(paste("Acurácia Random Forest (Teste): ", acuracia_rf))                               # "Acurácia Random Forest (Teste):  0.996"                      
rm(previsoes_rf)
rm(acuracia_rf)



## Modelo 3: SVM (Support Vector Machine)
modelo_svm <- svm(Class ~ ., data = dados_treino_fac)

# Faça previsões
previsoes_svm <- predict(modelo_svm, newdata = dados_teste_fac)

# Avalie a acurácia do modelo
acuracia_svm <- sum(previsoes_svm == dados_teste_fac$Class) / nrow(dados_teste_fac)
print(paste("Acurácia SVM (Teste): ", acuracia_svm))                                        # "Acurácia SVM (Teste):  0.976"
rm(previsoes_svm)
rm(acuracia_svm)


## -> Modelo RandomForest apresentou o melhor desempenho para prever a variável alvo Class
rm(modelo_arvore)
rm(modelo_svm)



str(dados_fac)
#### Respondendo Perguntas de Negócio

## 1) O modelo consegue identificar estratégias eficazes ou movimentos que frequentemente resultam em vitórias?

# Sim, o modelo RandomForest treinado foi capaz de identificar estratégias eficazes ou movimentos que frequentemente resultam em vitórias para 
# o jogador "X". A análise da importância das variáveis indicou que as posições específicas no tabuleiro, como middle_middle_square,
# top_left_square, top_right_square, bottom_left_square, e bottom_right_square, desempenham um papel crucial na previsão de vitórias para o 
# jogador "X". A alta acurácia do modelo no conjunto de teste (aproximadamente 98.95%) reforça sua capacidade de generalização e eficácia na 
# identificação dessas estratégias. Isso fornece insights valiosos para a equipe de design, permitindo melhorar o jogo da velha, destacando e
# enfatizando essas posições para criar uma experiência mais desafiadora e estratégica para os jogadores.



