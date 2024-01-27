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
library(shiny)          # interface gráfica
library(shinyjs)
library(shinyWidgets)
library(xgboost)        # carrega algoritimo de ML
library(Metrics)





#####################  Exemplo 1 (mesma lógica do projeto) (análise de dados / escolha modelo auto ml)


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


# Obter a importância das variáveis do modelo líder (Forma 2 os valores são obtidos diretamente do modelo líder)
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







##################### Exemplo 2  (análise de dados / escolha modelo auto ml / criacao do modelo em r / interface gráfica)

# História:

#  -> Uma empresa de telecomunicações está enfrentando um problema crescente de churn, onde clientes estão cancelando seus serviços.
#     A gerência está interessada em prever quais clientes têm maior probabilidade de cancelar, para que medidas proativas possam ser tomadas 
#     para retê-los.

# Perguntas de Negócio:

#  -> Quais fatores/métricas mais contribuem para a previsão de churn?
#  -> Qual é o melhor modelo de machine learning para prever o churn?
#  -> Como podemos explicar e interpretar as previsões do modelo de maneira compreensível para a equipe de negócios?



## Coleta e Análise dos Dados
dados_telecom <- tibble(
  tempo_assinatura = round(runif(3000, min = 1, max = 36), 0),                                  # Tempo de assinatura em meses
  tipo_plano = sample(c("Básico", "Intermediário", "Avançado"), 3000, replace = TRUE),          # Tipo de plano
  uso_servicos_adicionais = sample(c("Sim", "Não"), 3000, replace = TRUE, prob = c(0.4, 0.5)),  # Uso de serviços adicionais
  reclamacoes_recentes = sample(c("Sim", "Não"), 3000, replace = TRUE, prob = c(0.42, 0.58)),   # Reclamações recentes
  satisfacao_cliente = round(runif(3000, min = 1, max = 5), 1),                                 # Nível de satisfação do cliente
  gasto_mensal = rnorm(3000, mean = 100, sd = 20),                                              # Gasto mensal em dólares
  churn = sample(c("Sim", "Não"), 3000, replace = TRUE, prob = c(0.47, 0.53))                   # Variável alvo - Churn (cancelamento)
)



## Aplicando Eng. de Atributos e Adicionando Novas Variáveis 

# Tempo de Assinatura - Criar categorias
dados <- dados_telecom %>%
  mutate(tempo_assinatura_categoria = case_when(
    tempo_assinatura <= 12 ~ "Curto Prazo",
    tempo_assinatura <= 24 ~ "Médio Prazo",
    tempo_assinatura <= 36 ~ "Longo Prazo"
  ))

# Satisfação do Cliente - Criar categorias
dados <- dados %>%
  mutate(satisfacao_categoria = case_when(
    satisfacao_cliente <= 2 ~ "Baixa",
    satisfacao_cliente <= 4 ~ "Média",
    satisfacao_cliente <= 5 ~ "Alta"
  ))

# Gasto Mensal - Criar faixas
dados <- dados %>%
  mutate(gasto_mensal_categoria = cut(gasto_mensal, breaks = c(0, 50, 100, 150, 200), labels = c("0-50", "51-100", "101-150", "151-200")))



# Convertendo a variável alvo e qualquer variável chr para fator
dados <- dados %>% 
  mutate(churn = as.factor(churn)) %>%  
  mutate_if(is.character, factor)
str(dados)
summary(dados)



## Criando Modelo Para Seleção de Variáveis

# Criando Modelo Para Seleção De Variáveis
modelo <- randomForest(churn ~ ., data = dados, 
                       ntree = 100, nodesize = 10, importance = T)

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


rm(modelo)
rm(importancia_ordenada)
rm(df_importancia)
# -> Resposta: As variáveis com maior impacto no cancelamento são: tipo_plano, satisfacao_cliente, gasto_mensal, uso_servicos_adicionais e
#                                                                  reclamacoes_recentes



## Inicialização do h2o
h2o.init()

# O H2O requer que os dados estejam no formato de dataframe do H2O
dados <- dados %>% 
  select(tipo_plano, satisfacao_cliente, gasto_mensal, uso_servicos_adicionais, reclamacoes_recentes, churn)
h2o_frame <- as.h2o(dados)



## Divisão dos Dados em Treino e Teste

# Split dos dados em treino e teste (cria duas listas)
h2o_frame_split <- h2o.splitFrame(h2o_frame, ratios = 0.80)
head(h2o_frame_split)



## Modelo AutoML
modelo_automl <- h2o.automl(y = 'churn',
                            balance_classes = TRUE,
                            training_frame = h2o_frame_split[[1]],
                            nfolds = 4,
                            leaderboard_frame = h2o_frame_split[[2]],
                            max_runtime_secs = 60 * 2, 
                            include_algos = c('XGBoost', 'GBM', 'GLM'),
                            sort_metric = "AUC")

# Extrai o leaderboard (dataframe com os modelos criados)
leaderboard_automl <- as.data.frame(modelo_automl@leaderboard)
head(leaderboard_automl)
View(leaderboard_automl)

# Extrai o líder (modelo com melhor performance)
lider_automl <- modelo_automl@leader
print(lider_automl)
View(lider_automl)



## Avaliação do Modelo (Confusion Matrix)

# Avalie o desempenho do modelo no conjunto de teste
predicoes_teste <- h2o.predict(modelo_automl, newdata = h2o_frame_split[[2]])
predicoes_teste_df <- as.data.frame(h2o.cbind(h2o_frame_split[[2]], predicoes_teste))
confusion_matrix <- table(predicoes_teste_df$predict, predicoes_teste_df$churn)
print(confusion_matrix)

# Calcula a acurácia a partir da matriz de confusão
sum(diag(confusion_matrix)) / sum(confusion_matrix)    # 50%



## Plot da Importância das Variáveis

# Extraindo do melhor modelo a contribuição de cada variável para as previsões através dos dados de teste
# Estes valores são chamados de SHAP
var_contrib <- predict_contributions.H2OModel(lider_automl, h2o_frame_split[[2]])
var_contrib

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
df_var_contrib_gra <- df_var_contrib %>% 
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


## Respondendo as Perguntas de Negócio

# Quais fatores/métricas mais contribuem para a previsão de churn?
  
# -> De acordo com o modelo Random Forest e o modelo XGBoost (AutoML com SHAP), as variáveis mais importantes para prever o churn são:
#    Random Forest : Tipo de plano, Satisfação do cliente, Gasto mensal, Uso de serviços adicionais, Reclamações recentes.
#    XGBoost (SHAP): Gasto mensal, Satisfação do cliente, Uso de serviços adicionais, Reclamações recentes, Tipo de plano.

# Qual é o melhor modelo de machine learning para prever o churn?

# -> O melhor modelo de machine learning para prever o churn é o modelo líder do AutoML, que é um modelo XGBoost.

# Como podemos explicar e interpretar as previsões do modelo de maneira compreensível para a equipe de negócios?

# -> Utilizando a abordagem SHAP (SHapley Additive exPlanations), podemos explicar as previsões do modelo XGBoost da seguinte forma:
#    Gasto Mensal:
#      Maior contribuidor positivo para a previsão de churn. Aumentos no gasto mensal reduzem a probabilidade de churn.
#    Satisfação do Cliente:
#     Contribui positivamente para a previsão de churn. Clientes mais satisfeitos têm menor probabilidade de churn.
#    Uso de Serviços Adicionais:
#     Contribui positivamente, mas em menor medida.
#    Reclamações Recentes:
#     Contribui negativamente. Clientes com reclamações recentes têm maior probabilidade de churn.
#    Tipo de Plano:
#     Contribuição positiva, mas é menos influente que as variáveis mencionadas anteriormente.



## Salvando o modelo

# Salvar o modelo para um diretório específico
# h2o.saveModel(modelo_automl@leader, path = "modelo_exemplo2")


## Carregar o modelo a partir do diretório
h2o.init()
modelo_exemplo2 <- h2o.loadModel("modelos/modelo_exemplo2")
modelo_exemplo2

# Verificando variáveis usadas no modelo
importancia_variaveis <- h2o.varimp(modelo_exemplo2)
print(importancia_variaveis)
head(dados)

## Desliga o H2O
h2o.shutdown()
# write.csv(dados, file = "dados.csv", row.names = FALSE)


## Criando Modelo No R

## Carregando dados
dados <- read.csv("dados_exemplo2.csv")
str(dados)

# Convertendo a variável alvo e qualquer variável chr para fator
dados <- dados %>%
  mutate_if(is.character, factor)
str(dados)
summary(dados)

# Dividindo os dados em treino e teste
indices <- createDataPartition(dados$churn, p = 0.85, list = FALSE)
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)


# Criar a matriz DMatrix para o xgboost
dados_matrix_treino <- xgb.DMatrix(
  data = model.matrix(churn ~ . - 1, data = dados_treino),
  label = as.integer(dados_treino$churn) - 1
)
dados_matrix_teste <- xgb.DMatrix(
  data = model.matrix(churn ~ . - 1, data = dados_teste),
  label = as.integer(dados_teste$churn) - 1
)

## Criando Modelo

# Parâmetros do modelo (ajuste conforme necessário)
parametros_modelo <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.3,
  max_depth = 6,
  subsample = 1,
  colsample_bytree = 1,
  min_child_weight = 1
)

# Treinar o modelo XGBoost
modelo_xgboost_r <- xgboost(
  params = parametros_modelo,
  data = dados_matrix_treino,
  nrounds = 100
)

## Avaliar o Modelo

# Realizar previsões no conjunto de teste
previsoes_teste <- predict(modelo_xgboost_r, dados_matrix_teste)

# Converter as probabilidades em classes (1 ou 0)
previsoes_classes <- ifelse(previsoes_teste > 0.5, 1, 0)

# Matriz de Confusão
confusion_matrix <- table(previsoes_classes, dados_teste$churn)
print(confusion_matrix)

# Acurácia
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Acurácia:", accuracy))

# Verificar o comprimento dos vetores
length(previsoes_classes)
length(dados_teste$churn)



## Interface Gráfica

# Criar a matriz DMatrix para o xgboost
dados_matrix_treino <- xgb.DMatrix(
  data = model.matrix(churn ~ . - 1, data = dados_treino),
  label = as.integer(dados_treino$churn) - 1
)

ui <- fluidPage(
  titlePanel("Previsão de Churn"),
  sidebarLayout(
    sidebarPanel(
      pickerInput(
        "tipo_plano",
        label = "Tipo de Plano:",
        choices = unique(dados$tipo_plano),
        options = list('actions-box' = TRUE),
        multiple = FALSE
      ),
      sliderInput("satisfacao_cliente", "Satisfação do Cliente:", 1, 5, 3, step = 0.1),
      numericInput("gasto_mensal", "Gasto Mensal:", 100, min = 31.47, max = 167.37),
      selectInput("uso_servicos_adicionais", "Uso de Serviços Adicionais:", c("Não", "Sim")),
      selectInput("reclamacoes_recentes", "Reclamações Recentes:", c("Não", "Sim")),
      actionButton("verificar", "Verificar")
    ),
    mainPanel(
      h4("Resultado da Previsão:"),
      verbatimTextOutput("resultado")
    )
  )
)

# Definir a lógica do servidor
server <- function(input, output) {
  observeEvent(input$verificar, {
    # Criar um dataframe com os dados inseridos pelo usuário
    novo_dado <- data.frame(
      tipo_plano = factor(input$tipo_plano, levels = levels(dados$tipo_plano)),
      satisfacao_cliente = input$satisfacao_cliente,
      gasto_mensal = input$gasto_mensal,
      uso_servicos_adicionais = factor(input$uso_servicos_adicionais, levels = levels(dados$uso_servicos_adicionais)),
      reclamacoes_recentes = factor(input$reclamacoes_recentes, levels = levels(dados$reclamacoes_recentes))
    )
    
    # Ajustar o novo dado conforme necessário usando a fórmula do modelo treinado
    novo_dado <- model.matrix(~ . - 1, data = novo_dado)
    
    # Fazer a previsão
    previsao <- predict(modelo_xgboost_r, as.matrix(novo_dado), type = "response")
    
    # Exibir o resultado
    output$resultado <- renderText({
      if (previsao > 0.5) {
        return("Cliente provavelmente irá cancelar.")
      } else {
        return("Cliente provavelmente não irá cancelar.")
      }
    })
  })
}


# Criar o aplicativo Shiny
shinyApp(ui = ui, server = server)







##################### Exemplo 3  (análise de dados / escolha modelo auto ml / criacao do modelo em r / interface gráfica)

# História:

#  -> Uma plataforma de comércio eletrônico está buscando otimizar a experiência do usuário e aumentar as taxas de conversão.
#     A equipe de análise de dados identificou a necessidade de um sistema de recomendação personalizado para oferecer aos usuários
#     produtos relevantes, aumentando assim as chances de compra.

# Perguntas de Negócio:
  
# -> Quais são os principais fatores que influenciam a decisão de compra dos usuários?
#    Identificar as variáveis mais impactantes que contribuem para as decisões de compra dos usuários.

# -> Qual é o melhor modelo de sistema de recomendação para personalizar as sugestões de produtos?
#    Avaliar diferentes algoritmos de sistemas de recomendação, como filtragem colaborativa, filtragem baseada em conteúdo ou híbridos, 
#    para determinar o mais eficaz.

# -> Como podemos criar um sistema de recomendação interpretável e explicável para a equipe de marketing?
#    Desenvolver um modelo de interpretação para explicar as sugestões de produtos aos membros da equipe de marketing de maneira compreensível.

# -> Qual é o impacto esperado na taxa de conversão após a implementação do sistema de recomendação personalizado?
#    Avaliar o potencial aumento na taxa de conversão após a implementação do sistema de recomendação, medindo os resultados e comparando com
#    os dados históricos.

# -> Como podemos monitorar e ajustar continuamente o sistema de recomendação para manter a eficácia ao longo do tempo?
#    Estabelecer um processo de monitoramento contínuo para avaliar o desempenho do sistema de recomendação e implementar ajustes conforme
#    necessário.


##  Criar dados fictícios
dados <- data.frame(
  usuario_id = seq(1, 4000),
  idade = sample(18:65, 4000, replace = TRUE),
  genero = sample(c("Masculino", "Feminino"), 4000, replace = TRUE),
  historico_compras = sample(c("Baixo", "Médio", "Alto"), 4000, replace = TRUE, prob = c(0.4, 0.4, 0.2)),
  categoria_preferida = sample(c("Eletrônicos", "Moda", "Esportes", "Casa", "Beleza"), 4000, replace = TRUE),
  tempo_na_plataforma = sample(1:24, 4000, replace = TRUE),
  produtos_visualizados = rpois(4000, lambda = 20),
  nivel_atividade = sample(c("Baixo", "Médio", "Alto"), 4000, replace = TRUE, prob = c(0.3, 0.4, 0.3)),
  realizou_compra = factor(rbinom(4000, 1, 0.47), levels = c(0, 1), labels = c("Não", "Sim")),
  churn = factor(rbinom(4000, 1, 0.42), levels = c(0, 1), labels = c("Não", "Sim")),
  produto_recomendado = sample(c("A", "B", "C", "D", "E"), 4000, replace = TRUE)
)
str(dados)


## Salvando dataset
# write.csv(dados, file = "dados_exemplo3.csv", row.names = FALSE)



## Carregando dataset
dados <- read.csv("dados_exemplo3.csv")
str(dados)


## Análise Exploratória + Eng de Atributos

# Modificando qualquer variável chr para factor
dados <- dados %>% 
  mutate_if(is.character, factor)
str(dados)
summary(dados)
head(dados)

# Criação de Novas Características (Variáveis)
dados <- dados %>%
  mutate(
    razao_produtos_tempo = produtos_visualizados / tempo_na_plataforma,
    interacao_idade_produtos = idade * produtos_visualizados
  )
head(dados)
str(dados)
summary(dados)

# Escalonamento de Variáveis Numéricas (Aplicar somente para Modelo Criado no Ambiente R)
#dados <- dados %>%
#  select(-usuario_id) %>% 
#  mutate(across(where(is.numeric), scale))

# Remoção da Variável Id
dados <- dados %>%
  select(-usuario_id)

str(dados)
summary(dados)
head(dados)



#### Aplicando AutoML

## Inicialização do h2o
h2o.init()

# O H2O requer que os dados estejam no formato de dataframe do H2O
h2o_frame <- as.h2o(dados)


## Divisão dos Dados em Treino e Teste

# Split dos dados em treino e teste (cria duas listas)
h2o_frame_split <- h2o.splitFrame(h2o_frame, ratios = 0.85)
head(h2o_frame_split)
summary(h2o_frame_split)


## Modelos AutoML

# Modelo Realizou Compra
modelo_automl_rc <- h2o.automl(y = 'realizou_compra',
                               balance_classes = TRUE,
                               training_frame = h2o_frame_split[[1]],
                               nfolds = 4,
                               leaderboard_frame = h2o_frame_split[[2]],
                               max_runtime_secs = 60 * 2, 
                               include_algos = c('XGBoost', 'GBM', 'GLM'),
                               sort_metric = "AUC")

# Modelo Taxa de Cancelamento
modelo_automl_tc <- h2o.automl(y = 'churn',
                               balance_classes = TRUE,
                               training_frame = h2o_frame_split[[1]],
                               nfolds = 4,
                               leaderboard_frame = h2o_frame_split[[2]],
                               max_runtime_secs = 60 * 2, 
                               include_algos = c('XGBoost', 'GBM', 'GLM'),
                               sort_metric = "AUC")

# Modelo Produto Recomendado com Ajustes
modelo_automl_pr <- h2o.automl(y = 'produto_recomendado',
                               balance_classes = TRUE,
                               training_frame = h2o_frame_split[[1]],
                               nfolds = 4,
                               leaderboard_frame = h2o_frame_split[[2]],
                               max_runtime_secs = 60 * 22, 
                               sort_metric = "logloss",               # Use logloss para classificação multiclasse
                               exclude_algos = c("StackedEnsemble"))  # Excluir StackedEnsemble para simplificar


# Extrai o leaderboard (dataframe com os modelos criados)
leaderboard_automl_rc <- as.data.frame(modelo_rc@leaderboard)
leaderboard_automl_tc <- as.data.frame(modelo_automl_tc@leaderboard)
leaderboard_automl_pr <- as.data.frame(modelo_automl_pr@leaderboard)
head(leaderboard_automl_pr)
View(leaderboard_automl_pr)


# Extrai o líder (modelo com melhor performance)
lider_automl_rc <- modelo_automl_rc@leader
lider_automl_tc <- modelo_automl_tc@leader
lider_automl_pr <- modelo_automl_pr@leader
print(lider_automl_pr)
View(lider_automl_pr)


## Salvando Modelos (formato h2o e RDS)

# h2o.saveModel(lider_automl_rc, path = "modelos/modelo_automl_rc")
# saveRDS(lider_automl_rc, file = "modelos/modelo_automl_rc/XGBoost_R.rds")
# h2o.saveModel(lider_automl_tc, path = "modelos/modelo_automl_tc")
# saveRDS(lider_automl_tc, file = "modelos/modelo_automl_tc/XGBoost_R.rds")
# h2o.saveModel(lider_automl_pr, path = "modelos/modelo_automl_pr2")
# saveRDS(lider_automl_pr, file = "modelos/modelo_automl_pr/GLM_R.rds")

modelo_rc <- h2o.loadModel("modelos/modelo_automl_rc/XGBoost_AutoML")
modelo_rc

rm(leaderboard_automl_rc)
rm(leaderboard_automl_tc)
rm(leaderboard_automl_pr)



#### Avaliação dos Modelos

## Avaliação do Modelo Binomial (Realizou Compra)
perf_rc <- h2o.performance(modelo_rc)
perf_rc                                  # Verifica todas as métricas

# MSE                 :  0.02246815    -    Quanto mais próximo de zero, melhor. Valor encontrado: 0.02246815
# RMSE                :  0.1498938     -    Valores menores indicam melhor desempenho. Valor encontrado: 0.1498938
# LogLoss             :  0.1298117     -    Quanto mais próximo de zero, melhor. Valor encontrado: 0.1298117
# Mean Per-Class Error:  0.004263429   -    Quanto mais próximo de zero, melhor. Valor encontrado: 0.004263429
# AUC                 :  0.9999331     -    Um valor próximo de 1 indica bom desempenho. Valor encontrado: 0.9999331
# AUCPR               :  0.9999264     -    Um valor próximo de 1 indica bom desempenho. Valor encontrado: 0.9999264
# Gini                :  0.9998663
# R^2                 :  0.9099157


## Avaliação do Modelo Binomial (Taxa de Cancelamento)
perf_tc <- h2o.performance(modelo_tc)
perf_tc


## Avaliação do Modelo Deep Learning (Produto Recomendado)
perf_pr <- h2o.performance(modelo_pr)
perf_pr

head(dados)




## Respondendo Perguntas de Negócio

# Extraindo do melhor modelo a contribuição de cada variável para as previsões através dos dados de teste
# Estes valores são chamados de SHAP
var_contrib_rc <- predict_contributions.H2OModel(modelo_rc, h2o_frame_split[[2]])
var_contrib_tc <- predict_contributions.H2OModel(modelo_tc, h2o_frame_split[[2]])
var_contrib_tc

# Criando um dataframe com os as métricas que precisamos (Forma 1 de obter as variáveis mais importantes usando as métricas SHAP)
df_var_contrib_rc <- var_contrib_rc %>%
  as.data.frame() %>%
  select(-BiasTerm) %>%
  gather(feature, shap_value) %>%
  group_by(feature) %>%
  mutate(shap_importance = mean(abs(shap_value)), shap_force = mean(shap_value)) %>% 
  ungroup()
df_var_contrib_tc <- var_contrib_tc %>%
  as.data.frame() %>%
  select(-BiasTerm) %>%
  gather(feature, shap_value) %>%
  group_by(feature) %>%
  mutate(shap_importance = mean(abs(shap_value)), shap_force = mean(shap_value)) %>% 
  ungroup()
View(df_var_contrib_rc)

# Visualizando a média da importância de cada variável
importance_summary_rc <- df_var_contrib_rc %>%
  group_by(feature) %>%
  summarise(mean_shap_importance = mean(shap_importance),
            mean_shap_force = mean(shap_force)) %>% 
  arrange(-mean_shap_importance)
importance_summary_tc <- df_var_contrib_tc %>%
  group_by(feature) %>%
  summarise(mean_shap_importance = mean(shap_importance),
            mean_shap_force = mean(shap_force)) %>% 
  arrange(-mean_shap_importance)
print(importance_summary_rc)



## 1) Quais são os principais fatores que influenciam a decisão de compra dos usuários?
##    Identificar as variáveis mais impactantes que contribuem para as decisões de compra dos usuários.

# Gráfico de barras da importância de cada variável para prever a variável realizou compra
df_var_contrib_gra <- df_var_contrib_rc %>% 
  select(feature, shap_importance) %>%
  distinct() %>% 
  ggplot(aes(x = reorder(feature, shap_importance), y = shap_importance)) +
  geom_col(fill = 'blue') +
  coord_flip() +
  xlab(NULL) +
  ylab("Valor Médio das Métricas SHAP") +
  theme_minimal(base_size = 15)
df_var_contrib_gra


## 2) Qual é o melhor modelo de sistema de recomendação para personalizar as sugestões de produtos?
##    Avaliar diferentes algoritmos de sistemas de recomendação, como filtragem colaborativa, filtragem baseada em conteúdo ou híbridos, 
##    para determinar o mais eficaz.






## Desliga o H2O
h2o.shutdown()




#### Interface gráfica

## Inicialização do h2o
h2o.init()

## Carregando Modelos 
modelo_rc <- h2o.loadModel("modelos/modelo_automl_rc/XGBoost_AutoML")
modelo_tc <- h2o.loadModel("modelos/modelo_automl_tc/XGBoost")
modelo_pr <- h2o.loadModel("modelos/modelo_automl_pr/GLM_1_AutoML")


modelo_rc
modelo_tc
modelo_pr





## Interface Realizou Compra

# UI
ui <- fluidPage(
  titlePanel("Previsão de Compra - Modelo RC"),
  
  sidebarLayout(
    sidebarPanel(
      # Entradas do usuário
      numericInput("idade_input", "Idade:", min = 18, max = 100, value = 25),
      selectInput("genero_input", "Gênero:", choices = c("Feminino", "Masculino"), selected = "Feminino"),
      selectInput("historico_input", "Histórico de Compras:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
      selectInput("categoria_input", "Categoria Preferida:", choices = unique(dados$categoria_preferida), selected = unique(dados$categoria_preferida)[1]),
      numericInput("tempo_input", "Tempo na Plataforma (em meses):", min = 1, max = 100, value = 10),
      numericInput("produtos_input", "Produtos Visualizados:", min = 1, max = 100, value = 20),
      selectInput("atividade_input", "Nível de Atividade:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
      actionButton("previsao_btn", "Fazer Previsão")
    ),
    mainPanel(
      # Saída
      textOutput("resultado_texto")
    )
  )
)

# Server
server <- function(input, output) {
  
  observeEvent(input$previsao_btn, {
    # Criando um data frame temporário com os dados inseridos pelo usuário
    novo_dado <- data.frame(
      idade = input$idade_input,
      genero = input$genero_input,
      historico_compras = input$historico_input,
      categoria_preferida = input$categoria_input,
      tempo_na_plataforma = input$tempo_input,
      produtos_visualizados = input$produtos_input,
      nivel_atividade = input$atividade_input
    )
    
    # Modificando qualquer variável chr para factor
    novo_dado <- novo_dado %>% mutate_if(is.character, factor)
    
    # Criação de Novas Características (Variáveis)
    novo_dado <- novo_dado %>% mutate(
      razao_produtos_tempo = produtos_visualizados / tempo_na_plataforma,
      interacao_idade_produtos = idade * produtos_visualizados
    )
    
    # Adicionando colunas temporárias faltantes
    novo_dado$produto_recomendado <- NA
    novo_dado$churn <- NA
    
    # Fazendo a previsão
    previsao <- h2o.predict(modelo_rc, as.h2o(novo_dado))
    
    # Mensagem de saída
    output$resultado_texto <- renderText({
      if (previsao[1, "predict"] == "Sim") {
        "SIM. Há uma grande possibilidade do usuário realizar a compra."
      } else {
        "NÃO. Há uma grande possibilidade do usuário não realizar a compra."
      }
    })
  })
}

# Rodando a aplicação
shinyApp(ui = ui, server = server)



## Interface Churn

# UI
ui <- fluidPage(
  titlePanel("Previsão de Churn - Modelo TC"),
  
  sidebarLayout(
    sidebarPanel(
      # Entradas do usuário
      numericInput("idade_input", "Idade:", min = 18, max = 100, value = 25),
      selectInput("genero_input", "Gênero:", choices = c("Feminino", "Masculino"), selected = "Feminino"),
      selectInput("historico_input", "Histórico de Compras:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
      selectInput("categoria_input", "Categoria Preferida:", choices = unique(dados$categoria_preferida), selected = unique(dados$categoria_preferida)[1]),
      numericInput("tempo_input", "Tempo na Plataforma (em meses):", min = 1, max = 100, value = 10),
      numericInput("produtos_input", "Produtos Visualizados:", min = 1, max = 100, value = 20),
      selectInput("atividade_input", "Nível de Atividade:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
      actionButton("previsao_btn", "Fazer Previsão")
    ),
    mainPanel(
      # Saída
      textOutput("resultado_texto")
    )
  )
)

# Server
server <- function(input, output) {
  
  observeEvent(input$previsao_btn, {
    # Criando um data frame temporário com os dados inseridos pelo usuário
    novo_dado <- data.frame(
      idade = input$idade_input,
      genero = input$genero_input,
      historico_compras = input$historico_input,
      categoria_preferida = input$categoria_input,
      tempo_na_plataforma = input$tempo_input,
      produtos_visualizados = input$produtos_input,
      nivel_atividade = input$atividade_input
    )
    
    # Modificando qualquer variável chr para factor
    novo_dado <- novo_dado %>% mutate_if(is.character, factor)
    
    # Criação de Novas Características (Variáveis)
    novo_dado <- novo_dado %>% mutate(
      razao_produtos_tempo = produtos_visualizados / tempo_na_plataforma,
      interacao_idade_produtos = idade * produtos_visualizados
    )
    
    # Adicionando coluna temporária faltante
    novo_dado$realizou_compra <- NA
    novo_dado$produto_recomendado <- NA
    
    # Fazendo a previsão
    previsao <- h2o.predict(modelo_tc, as.h2o(novo_dado))
    
    # Mensagem de saída
    output$resultado_texto <- renderText({
      if (previsao[1, "predict"] == "Sim") {
        "SIM. Há uma grande possibilidade do usuário cancelar o serviço."
      } else {
        "NÃO. Há uma grande possibilidade do usuário não cancelar o serviço."
      }
    })
  })
}

# Rodando a aplicação
shinyApp(ui = ui, server = server)



## Interface Produtos Recomendados

# UI
ui <- fluidPage(
  titlePanel("Recomendação de Produtos - Modelo PR"),
  
  sidebarLayout(
    sidebarPanel(
      # Entradas do usuário
      numericInput("idade_input", "Idade:", min = 18, max = 100, value = 25),
      selectInput("genero_input", "Gênero:", choices = c("Feminino", "Masculino"), selected = "Feminino"),
      selectInput("historico_input", "Histórico de Compras:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
      selectInput("categoria_input", "Categoria Preferida:", choices = unique(dados$categoria_preferida), selected = unique(dados$categoria_preferida)[1]),
      numericInput("tempo_input", "Tempo na Plataforma (em meses):", min = 1, max = 100, value = 10),
      numericInput("produtos_input", "Produtos Visualizados:", min = 1, max = 100, value = 20),
      selectInput("atividade_input", "Nível de Atividade:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
      actionButton("recomendacao_btn", "Recomendar Produtos")
    ),
    mainPanel(
      # Saída
      textOutput("resultado_texto")
    )
  )
)

# Server
server <- function(input, output) {
  
  observeEvent(input$recomendacao_btn, {
    # Criando um data frame temporário com os dados inseridos pelo usuário
    novo_dado <- data.frame(
      idade = input$idade_input,
      genero = input$genero_input,
      historico_compras = input$historico_input,
      categoria_preferida = input$categoria_input,
      tempo_na_plataforma = input$tempo_input,
      produtos_visualizados = input$produtos_input,
      nivel_atividade = input$atividade_input
    )
    
    # Modificando qualquer variável chr para factor
    novo_dado <- novo_dado %>% mutate_if(is.character, factor)
    
    # Criação de Novas Características (Variáveis)
    novo_dado <- novo_dado %>% mutate(
      razao_produtos_tempo = produtos_visualizados / tempo_na_plataforma,
      interacao_idade_produtos = idade * produtos_visualizados
    )
    
    # Adicionando colunas temporárias faltantes
    novo_dado$realizou_compra <- NA
    novo_dado$churn <- NA
    
    # Fazendo a previsão
    previsao <- h2o.predict(modelo_pr, as.h2o(novo_dado))
    
    # Mensagem de saída
    output$resultado_texto <- renderText({
      paste("Produto Recomendado:", previsao[1, "predict"])
    })
  })
}

# Rodando a aplicação
shinyApp(ui = ui, server = server)




### Interface 3 Modelos

library(shiny)
library(dplyr)

# Função para realizar a previsão com os modelos
realizar_previsao <- function(modelo, novo_dado) {
  # Modificando qualquer variável chr para factor
  novo_dado <- novo_dado %>% mutate_if(is.character, factor)
  
  # Criação de Novas Características (Variáveis)
  novo_dado <- novo_dado %>% mutate(
    razao_produtos_tempo = produtos_visualizados / tempo_na_plataforma,
    interacao_idade_produtos = idade * produtos_visualizados
  )
  
  # Adicionando colunas temporárias faltantes
  if (!("realizou_compra" %in% colnames(novo_dado))) novo_dado$realizou_compra <- NA
  if (!("churn" %in% colnames(novo_dado))) novo_dado$churn <- NA
  if (!("produto_recomendado" %in% colnames(novo_dado))) novo_dado$produto_recomendado <- NA
  
  # Fazendo a previsão
  previsao <- h2o.predict(modelo, as.h2o(novo_dado))
  
  return(previsao)
}

# UI principal
ui <- fluidPage(
  titlePanel("Escolha do Modelo"),
  
  sidebarLayout(
    sidebarPanel(
      # Escolha do modelo
      selectInput("modelo_input", "Escolha o Modelo:",
                  choices = c("PrevisaoCompra", "PrevisaoChurn", "RecomendacaoProdutos")),
    ),
    mainPanel(
      # Conteúdo da interface do modelo escolhido
      uiOutput("modelo_ui"),
      # Saída
      textOutput("resultado_texto")
    )
  )
)

# Server principal
server <- function(input, output, session) {
  
  # Função para criar a interface do modelo escolhido
  output$modelo_ui <- renderUI({
    modelo_escolhido <- input$modelo_input
    
    # Criar a interface do modelo escolhido
    if (!is.null(modelo_escolhido)) {
      
      if (modelo_escolhido == "PrevisaoCompra") {
        # Interface do modelo PrevisaoCompra
        fluidPage(
          titlePanel("Previsão de Compra - Modelo RC"),
          numericInput("idade_input", "Idade:", min = 18, max = 100, value = 25),
          selectInput("genero_input", "Gênero:", choices = c("Feminino", "Masculino"), selected = "Feminino"),
          selectInput("historico_input", "Histórico de Compras:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
          selectInput("categoria_input", "Categoria Preferida:", choices = unique(dados$categoria_preferida), selected = unique(dados$categoria_preferida)[1]),
          numericInput("tempo_input", "Tempo na Plataforma (em meses):", min = 1, max = 100, value = 10),
          numericInput("produtos_input", "Produtos Visualizados:", min = 1, max = 100, value = 20),
          selectInput("atividade_input", "Nível de Atividade:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
          actionButton("previsao_btn", "Fazer Previsão")
        )
        
      } else if (modelo_escolhido == "PrevisaoChurn") {
        # Interface do modelo PrevisaoChurn
        fluidPage(
          titlePanel("Previsão de Churn - Modelo TC"),
          numericInput("idade_input", "Idade:", min = 18, max = 100, value = 25),
          selectInput("genero_input", "Gênero:", choices = c("Feminino", "Masculino"), selected = "Feminino"),
          selectInput("historico_input", "Histórico de Compras:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
          selectInput("categoria_input", "Categoria Preferida:", choices = unique(dados$categoria_preferida), selected = unique(dados$categoria_preferida)[1]),
          numericInput("tempo_input", "Tempo na Plataforma (em meses):", min = 1, max = 100, value = 10),
          numericInput("produtos_input", "Produtos Visualizados:", min = 1, max = 100, value = 20),
          selectInput("atividade_input", "Nível de Atividade:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
          actionButton("previsao_btn", "Fazer Previsão")
        )
        
      } else if (modelo_escolhido == "RecomendacaoProdutos") {
        # Interface do modelo RecomendacaoProdutos
        fluidPage(
          titlePanel("Recomendação de Produtos - Modelo PR"),
          numericInput("idade_input", "Idade:", min = 18, max = 100, value = 25),
          selectInput("genero_input", "Gênero:", choices = c("Feminino", "Masculino"), selected = "Feminino"),
          selectInput("historico_input", "Histórico de Compras:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
          selectInput("categoria_input", "Categoria Preferida:", choices = unique(dados$categoria_preferida), selected = unique(dados$categoria_preferida)[1]),
          numericInput("tempo_input", "Tempo na Plataforma (em meses):", min = 1, max = 100, value = 10),
          numericInput("produtos_input", "Produtos Visualizados:", min = 1, max = 100, value = 20),
          selectInput("atividade_input", "Nível de Atividade:", choices = c("Alto", "Baixo", "Médio"), selected = "Alto"),
          actionButton("recomendacao_btn", "Recomendar Produtos")
        )
      }
    }
  })
  
  # Lógica para realizar a previsão
  observeEvent(input$previsao_btn, {
    # Criando um data frame temporário com os dados inseridos pelo usuário
    novo_dado <- data.frame(
      idade = input$idade_input,
      genero = input$genero_input,
      historico_compras = input$historico_input,
      categoria_preferida = input$categoria_input,
      tempo_na_plataforma = input$tempo_input,
      produtos_visualizados = input$produtos_input,
      nivel_atividade = input$atividade_input
    )
    
    # Realizar previsão com base no modelo escolhido
    previsao <- switch(
      input$modelo_input,
      PrevisaoCompra = realizar_previsao(modelo_rc, novo_dado),
      PrevisaoChurn = realizar_previsao(modelo_tc, novo_dado),
      RecomendacaoProdutos = realizar_previsao(modelo_pr, novo_dado)
    )
    
    # Mensagem de saída
    output$resultado_texto <- renderText({
      if (previsao[1, "predict"] == "Sim") {
        "SIM. Há uma grande possibilidade do usuário realizar a ação desejada."
      } else {
        "NÃO. Há uma grande possibilidade do usuário não realizar a ação desejada."
      }
    })
  })
  
  # Lógica específica para RecomendacaoProdutos
  observeEvent(input$recomendacao_btn, {
    # Criando um data frame temporário com os dados inseridos pelo usuário
    novo_dado <- data.frame(
      idade = input$idade_input,
      genero = input$genero_input,
      historico_compras = input$historico_input,
      categoria_preferida = input$categoria_input,
      tempo_na_plataforma = input$tempo_input,
      produtos_visualizados = input$produtos_input,
      nivel_atividade = input$atividade_input
    )
    
    # Realizar previsão com base no modelo escolhido
    previsao <- realizar_previsao(modelo_pr, novo_dado)
    
    # Mensagem de saída
    output$resultado_texto <- renderText({
      paste("Produto Recomendado:", previsao[1, "predict"])
    })
  })
}

# Rodando a aplicação
shinyApp(ui = ui, server = server)












## Desliga o H2O
h2o.shutdown()







head(dados)
str(dados)
h2o.varimp(modelo_rc)
h2o.varimp(modelo_tc)
h2o.varimp(modelo_pr)
