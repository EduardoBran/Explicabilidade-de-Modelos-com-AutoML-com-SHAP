# Mini-Projeto 3 - Explicabilidade de Modelos AutoML com SHAP (SHapley Additive exPlanations)

# Leia os manuais em pdf no Capítulo 19 do curso.

# Instalação dos pacotes
install.packages("h2o")
install.packages("tidyverse")
install.packages("ggbeeswarm")

# Carrega os pacotes na sessão
library(h2o)
library(tidyverse)
library(ggbeeswarm)

# Preparação da massa de dados
# Como os dados serão gerados de forma randômica, o resultado será diferente a cada execução
dataset <- tibble(produtividade = c(rnorm(1000), rnorm(1000, 0.25)),
                  rendimento = runif(2000),
                  custo = rf(2000, df1 = 5, df2 = 2),
                  prioridade = c(sample(rep(c('Baixa', 'Media', 'Alta'), c(300, 300, 400))), 
                                 sample(c('Baixa', 'Media', 'Alta'), 1000, prob = c(0.25, 0.25, 0.5), replace = T)),
                  eficiencia = rnorm(2000),
                  manutencao = rep(c(0,1), c(1050,950)))

# Dimensões
dim(dataset)

# Visualiza os dados
View(dataset)

# Tipos dos dados
str(dataset)

# A variável alvo é "manutencao"
table(dataset$manutencao)

# A variável 4 é categórica
table(dataset$prioridade)

# Vamos converter a variável alvo para o tipo fator
# Isso é requerido pelo H2O
# A variável preditora categórica também será convertida
dataset <- dataset %>% 
  mutate(manutencao = as.factor(manutencao)) %>% 
  mutate_if(is.character, factor)

# Tipos dos dados
str(dataset)

# Visualiza os dados
View(dataset)

# Inicializamos o H2O (Framework de Machine Learning)
# Atenção à versão do Java JDK. Instale a versão 11 a partir do link abaixo:
# https://www.oracle.com/java/technologies/downloads/
h2o.init()

# O H2O requer que os dados estejam no formato de dataframe do H2O
h2o_frame <- as.h2o(dataset)
class(h2o_frame)
head(h2o_frame)

# Split dos dados em treino e teste
?h2o.splitFrame
h2o_frame_split <- h2o.splitFrame(h2o_frame, ratios = 0.77)
head(h2o_frame_split)

# Modelo AutoML
?h2o.automl
modelo_automl <- h2o.automl(y = 'manutencao',
                            balance_classes = TRUE,
                            training_frame = h2o_frame_split[[1]],
                            nfolds = 4,
                            leaderboard_frame = h2o_frame_split[[2]],
                            max_runtime_secs = 60 * 2, 
                            include_algos = c('XGBoost', 'GBM', 'GLM'),
                            sort_metric = "AUC")

# Extrai o leaderboard
leaderboard_automl <- as.data.frame(modelo_automl@leaderboard)
View(leaderboard_automl)

# Extrai o líder (modelo com melhor performance)
lider_automl <- modelo_automl@leader
View(lider_automl)

# Para o melhor modelo extraímos a contribuição de cada variável para as previsões
# os valores extraídos são chamados de valores SHAP
# Usamos os dados de teste
?predict_contributions.H2OModel
var_contrib <- predict_contributions.H2OModel(lider_automl, h2o_frame_split[[2]])

# Vamos visualizar o resultado final

# Primeiro preparamos um dataframe com os as métricas que precisamos
df_var_contrib <- var_contrib %>%
  as.data.frame() %>%
  select(-BiasTerm) %>%
  gather(feature, shap_value) %>%
  group_by(feature) %>%
  mutate(shap_importance = mean(abs(shap_value)), shap_force = mean(shap_value)) %>% 
  ungroup()

View(df_var_contrib)

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

# Plot de contribuição de cada variável para explicar a variável alvo
ggplot(df_var_contrib, aes(x = shap_value, y = reorder(feature, shap_importance))) +
  ggbeeswarm::geom_quasirandom(groupOnX = FALSE, varwidth = TRUE, size = 0.9, alpha = 0.5, width = 0.15) +
  xlab("Contribuição da Variável") +
  ylab(NULL) +
  theme_minimal(base_size = 15)

# Desliga o H2O
h2o.shutdown()



