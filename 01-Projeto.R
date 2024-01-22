####  Objetivo e Definição do Problema de Negócio  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/19.Mini-Projeto-3_-_Explicabilidade_de_Modelos_AutoML_com_SHAP")
getwd()


## Carregando pacotes
library(h2o)
library(tidyverse)
library(ggbeeswarm)
library(dplyr)
library(randomForest)
library(ROCR)  # Gerando uma curva ROC em R
library(caret) # Cria confusion matrix


# -> O objetivo final deste mini projeto não consiste em estudar a fundo o AutoML, o objetivo real é mostrar o processo completo desde a definição
#    do problema de negócio até a entrega/explicabilidade. Além da criação do modelo, também daremos mais um passo explicando como o modelo faz
#    as previsões.



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


## Análise Exploratória

# Visualização dos dados
dim(dados)
str(dados)
table(dados$manutencao)

# Modificando variáveis prioridade e manutencao para tipo factor (é obrigatório a variável alvo está tipo factor para uso do h2o)
dados <- dados %>% 
  mutate(manutencao = as.factor(manutencao)) %>% 
  mutate_if(is.character, factor) # modifica todas as variáveis do tipo chr para factor
str(dados)





## Criando Modelo Preditivo de forma tradicional (sem AutoML)

# Seleção de Variáveis (Feature Selection)
modelo <- randomForest(manutencao ~ ., data = dados, 
                       ntree = 100, nodesize = 10, importance = T)

# Visualizando por números
print(modelo$importance)
varImpPlot(modelo)

# Visualizando (usando ggplot, método mais profissional)
importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 

df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))

rm(importancia_ordenada)
rm(df_importancia)


## Criação do Modelo Preditivo

# Dividindo os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados$manutencao, p = 0.90, list = FALSE)  
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)

# Modelo
modelo <- randomForest(manutencao ~ ., data = dados_treino, ntree = 100, nodesize = 10)
modelo

# Gerando previsões nos dados de teste
previsoes <- data.frame(observado = dados_teste$manutencao,
                        previsto = predict(modelo, newdata = dados_teste))

# Confusion Matrix (utilizando pacote Caret)
confusionMatrix(previsoes$observado, previsoes$previsto)  # Accuracy : 0.505 


## Gerando Curva ROC

# Gerando as classes de dados
class1 <- predict(modelo, newdata = dados_teste, type = 'prob')
class2 <- dados_teste$manutencao

# Criando curva
pred <- prediction(class1[,2], class2)
perf <- performance(pred, "tpr","fpr") 
plot(perf, col = rainbow(10))

rm(class1)
rm(class2)
rm(pred)
rm(perf)
