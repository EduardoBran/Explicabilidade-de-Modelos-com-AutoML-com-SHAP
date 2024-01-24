####  Projeto  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/19.Mini-Projeto-3_-_Explicabilidade_de_Modelos_AutoML_com_SHAP")
getwd()


## Carregando pacotes 
library(h2o)            # framework para construir modelos de machine learning
library(tidyverse)      # manipulação de dados
library(ggbeeswarm)     # pacote que permite criar gráficos customizados sobre o ggplot2
library(ggplot2)        # gera gráficos
library(dplyr)          # manipulação de dados
library(randomForest)   # carrega algoritimo de ML
library(ROCR)           # Gerando uma curva ROC em R
library(caret)          # Cria confusion matrix
        



# -> O objetivo final deste mini projeto não consiste em estudar a fundo o AutoML, o objetivo real é mostrar o processo completo desde a definição
#    do problema de negócio até a entrega/explicabilidade. Além da criação do modelo, também daremos mais um passo explicando como o modelo faz
#    as previsões.


# -> Com o AutoML conseguiremos com uma única linha de código criar dezenas de modelos de Machine Learning.
#    Após isso iremos escolher o melhor modelo e então realizaremos uma Análise de Explicabilidade e entregar a Gerência como o modelo
#    chega ao seu resultado e sua conclusão. 




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


# -> Pergunta de negócio: Quais fatores/métricas (variáveis) mais contribuem para explicar o comportamento da variável alvo? Por que?
#    Variável alvo      : manutencao



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



## Inicializando o H2O (Framework de Machine Learning)

#  -> H20 é um framework de Machine Learning distribuído, no seu ambiente R.
#     O H2O é executado em uma máquina virtual Java (JVM) e fornece uma interface amigável para treinar modelos de Machine Learning.
h2o.init()

#  -> Após inicializado, será criado uma espécie de página web na porta:  H2O Connection port: 54321
#     http://localhost:54321/


# O H2O requer que os dados estejam no formato de dataframe do H2O
h2o_frame <- as.h2o(dados)
class(h2o_frame)
head(h2o_frame)


# Split dos dados em treino e teste (cria duas listas)
h2o_frame_split <- h2o.splitFrame(h2o_frame, ratios = 0.77)
head(h2o_frame_split)


# Modelo AutoML
modelo_automl <- h2o.automl(y = 'manutencao',
                            balance_classes = TRUE,
                            training_frame = h2o_frame_split[[1]],
                            nfolds = 4,
                            leaderboard_frame = h2o_frame_split[[2]],
                            max_runtime_secs = 60 * 2, 
                            include_algos = c('XGBoost', 'GBM', 'GLM'),
                            sort_metric = "AUC")

# y                 - variável alvo
# balance_classes   - informar se a classe alvo está balanceada (no nosso dados já está)
# training_frame    - dataset de treino
# nfolds            - técnica de validação cruzada para obter mais performance
# leaderboard_frame - dataset de teste
# max_runtime_secs  - tempo máximo de execução
# include_algos     - incluí os algoritmos escolhidos (se não colocar, ele testa com todos)
# sort_metric       - métrica usada para a ordenação dos melhores modelos


# Extrai o leaderboard (dataframe com os modelos criados)
leaderboard_automl <- as.data.frame(modelo_automl@leaderboard)
head(leaderboard_automl)
View(leaderboard_automl)

# Extrai o líder (modelo com melhor performance)
lider_automl <- modelo_automl@leader
print(lider_automl)
View(lider_automl)


# Extraindo do melhor modelo a contribuição de cada variável para as previsões através dos dados de teste
# Estes valores são chamados de SHAP
var_contrib <- predict_contributions.H2OModel(lider_automl, h2o_frame_split[[2]])
var_contrib




## Visualizando o Resultado Final

# Forma 1 de visualizar e obter resultados das variáveis mais importantes (usando métricas SHAP)
df_var_contrib <- var_contrib %>%
  as.data.frame() %>%
  select(-BiasTerm) %>%
  gather(feature, shap_value) %>%
  group_by(feature) %>%
  mutate(shap_importance = mean(abs(shap_value)), shap_force = mean(shap_value)) %>% 
  ungroup()
View(df_var_contrib)
head(df_var_contrib)


# Plot da importância de cada variável para prever a variável alvo
df_var_contrib %>% 
  select(feature, shap_importance) %>%
  distinct() %>% 
  ggplot(aes(x = reorder(feature, shap_importance), y = shap_importance)) +
  geom_col(fill = 'blue') +
  coord_flip() +
  xlab(NULL) +
  ylab("Valor Médio das Métricas SHAP") +
  theme_minimal(base_size = 15)

# O gráfico acima mostra a distribuição de todos os valores de SHAP para cada recurso no conjunto de teste. Pode-se observar que a 
# variável produtividade tem a maioria dos valores de Shapley positivos e tem uma distribuição mais ampla indicando sua importância
# no poder preditivo do modelo, enquanto a variável custo é a "menos importante". Isso também é reforçado pelo gráfico de importância 
# dado pela média de todos os valores absolutos das métricas.


# Plot de contribuição de cada variável para explicar a variável alvo
ggplot(df_var_contrib, aes(x = shap_value, y = reorder(feature, shap_importance))) +
  ggbeeswarm::geom_quasirandom(groupOnX = FALSE, varwidth = TRUE, size = 0.9, alpha = 0.5, width = 0.15) +
  xlab("Contribuição da Variável") +
  ylab(NULL) +
  theme_minimal(base_size = 15)

# Logo, à medida que aumenta o valor SHAP da variável produtividade, aumenta a probabilidade de previsão positiva do modelo (classe 1).
# O mesmo raciocínio se aplica para as variáveis prioridade e rendimento, porém em menor proporção. O modelo entende que aumento de
# produtividade, maior prioridade e aumento do rendimento levam à necessidade de manutenção.

# Já para a variável eficiencia, a maioria dos valores SHAP é negativa, o que aumenta a probabilidade de previsão negativa do 
# modelo (Classe 0). Nesse caso, o modelo entende que se a máquina tem menor eficiência, ela está sendo menos usada e, consequentemente,
# não requer manutenção no período avaliado.

# Para a variável custo a maioria dos valores está próximo de zero, indicando que a variável não é relevante para as previsões
# do modelo e poderia inclusive ser descartada para a versão final do modelo.



# Forma 2 de visualizar e obter resultados das variáveis mais importantes (os valores são obtidos diretamente do modelo líder GBM)
# (os valores são obtidos diretamente do modelo líder XGBoost líder e representam a importância relativa de cada variável de acordo com o algoritmo específico utilizado (XGBoost, neste caso).
importancia_variaveis <- h2o.varimp(modelo_automl@leader)
importancia_variaveis

# Converter para um dataframe do R
variaveis_importantes <- as.data.frame(importancia_variaveis)
variaveis_importantes

# Selecionar as variáveis mais importantes
variaveis_importantes <- variaveis_importantes %>%
  arrange(desc(relative_importance)) %>%
  select(variable, relative_importance)

# Visualizar as variáveis mais importantes
print(variaveis_importantes)

# Crie um gráfico de barras para visualizar a importância das variáveis
ggplot(variaveis_importantes, aes(x = reorder(variable, relative_importance), y = relative_importance)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Importância das Variáveis no Modelo",
       x = "Variáveis",
       y = "Importância") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))





# Desliga o H2O
h2o.shutdown()




















## Criando Modelo Preditivo de Classificação de forma tradicional (sem AutoML)

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
