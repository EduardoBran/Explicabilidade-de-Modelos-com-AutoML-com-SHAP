####  Projeto  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/19.Mini-Projeto-3_-_Explicabilidade_de_Modelos_AutoML_com_SHAP")
getwd()


## Carregar Pacotes
library(h2o)            # framework para construir modelos de machine learning
library(tidyverse)      # manipulação de dados
library(ggbeeswarm)     # pacote que permite criar gráficos customizados sobre o ggplot2
library(ggplot2)        # gera gráficos
library(dplyr)          # manipulação de dados
library(randomForest)   # carrega algoritimo de ML
library(ROCR)           # Gerando uma curva ROC em R
library(caret)          # Cria confusion matrix
library(shapper)



#### Exemplo 1 (mesma lógica do projeto)


## Etapa 1: Criação dos Dados Fictícios
dados <- tibble(produtividade = c(rnorm(1000), rnorm(1000, 0.25)),
                rendimento = runif(2000),
                custo = rf(2000, df1 = 5, df2 = 2),
                prioridade = c(sample(rep(c('Baixa', 'Media', 'Alta'), c(300, 300, 400))), 
                               sample(c('Baixa', 'Media', 'Alta'), 1000, prob = c(0.25, 0.25, 0.5), replace = T)),
                eficiencia = rnorm(2000),
                manutencao = rep(c(0,1), c(1050,950)))

# Modificando variáveis prioridade e manutencao para tipo factor (é obrigatório a variável alvo está tipo factor para uso do h2o)
dados <- dados %>% 
  mutate(manutencao = as.factor(manutencao)) %>% 
  mutate_if(is.character, factor) # modifica todas as variáveis do tipo chr para factor
str(dados)
head(dados)


## Etapa 2: Inicialização do h2o
h2o.init()



## Etapa 3: Divisão dos Dados em Treino e Teste

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


## Etapa 5: Avaliação do Modelo

# Avalie o desempenho do modelo no conjunto de teste
predicoes_teste <- h2o.predict(modelo_automl, newdata = h2o_frame_split[[2]])
predicoes_teste_df <- as.data.frame(h2o.cbind(h2o_frame_split[[2]], predicoes_teste))
confusion_matrix <- table(predicoes_teste_df$predict, predicoes_teste_df$manutencao)
print(confusion_matrix)

#       0   1
#   0  16  15
#   1 144 133

# O modelo previu corretamente 133 instâncias que realmente pertencem à classe "manutenção" (1).
# O modelo previu corretamente 16 instâncias que realmente pertencem à classe "sem manutenção" (0).
# O modelo previu incorretamente 15 instâncias como pertencentes à classe "manutenção".
# O modelo previu incorretamente 144 instâncias como pertencentes à classe "sem manutenção".

# Calcula a acurácia a partir da matriz de confusão
sum(diag(confusion_matrix)) / sum(confusion_matrix)    # 48%



## Etapa Final: Plot da Importância das Variáveis

# Criando um dataframe com os as métricas que precisamos (Forma 1 de obter as variáveis mais importantes usando as métricas SHAP)
df_var_contrib <- var_contrib %>%
  as.data.frame() %>%
  select(-BiasTerm) %>%
  gather(feature, shap_value) %>%
  group_by(feature) %>%
  mutate(shap_importance = mean(abs(shap_value)), shap_force = mean(shap_value)) %>% 
  ungroup()
View(df_var_contrib)
head(df_var_contrib)

# Visualizando a média da importância de cada variável
importance_summary <- df_var_contrib %>%
  group_by(feature) %>%
  summarise(mean_shap_importance = mean(shap_importance),
            mean_shap_force = mean(shap_force)) %>% 
  arrange(-mean_shap_importance)
print(importance_summary)


# Gráfico de barras da importância de cada variável para prever a variável alvo
df_var_contrib %>% 
  select(feature, shap_importance) %>%
  distinct() %>% 
  ggplot(aes(x = reorder(feature, shap_importance), y = shap_importance)) +
  geom_col(fill = 'blue') +
  coord_flip() +
  xlab(NULL) +
  ylab("Valor Médio das Métricas SHAP") +
  theme_minimal(base_size = 15)

# Gráfico de pares (pair plot) de cada variável para explicar a variável alvo
ggplot(df_var_contrib, aes(x = shap_value, y = reorder(feature, shap_importance))) +
  ggbeeswarm::geom_quasirandom(groupOnX = FALSE, varwidth = TRUE, size = 0.9, alpha = 0.5, width = 0.15) +
  xlab("Contribuição da Variável") +
  ylab(NULL) +
  theme_minimal(base_size = 15)

# Boxplot dos SHAP values por variável
ggplot(df_var_contrib, aes(x = feature, y = shap_value)) +
  geom_boxplot(fill = 'blue', alpha = 0.5) +
  coord_flip() +
  xlab(NULL) +
  ylab("SHAP Value") +
  theme_minimal(base_size = 15)

# Gráfico de densidade dos SHAP values por variável
ggplot(df_var_contrib, aes(x = shap_value, fill = feature)) +
  geom_density(alpha = 0.5) +
  xlab("SHAP Value") +
  ylab(NULL) +
  theme_minimal(base_size = 15)


# Obter a importância das variáveis do modelo líder (Forma 2 os valores são obtidos diretamente do modelo XGBoost)
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










## Desliga o H2O
h2o.shutdown()
