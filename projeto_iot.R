#PROJETO DE PREVISÃO DE USO ENERGÉTICO RESIDENCIAL (IoT)

#Projeto da Data Science Academy no qual há um feedback para avaliação de aprendizado


#Setando o diretório do projeto
setwd("machine_learning/Datasets e projetos/Projetos-7-8/Projeto 8/Modelagem_Preditiva_em_IoT/")
getwd()

#---------------------------CARREGANDO OS PACOTES------------------------------#
#carregando os pacotes a serem utilizados

library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(lubridate)
library(party)
library(randomForest)
library(Metrics)
library(CatEncoders)
library(caret)
library(h2o)
library(performance)


#---------------------------CARREGANDO OS DADOS--------------------------------#

#Os dados fornecidos já estão divididos entre treino e teste:

df_treino <- read.csv("projeto8-data_files/projeto8-training.csv") 
df_teste <- read.csv("projeto8-data_files/projeto8-testing.csv")


#Verificando o dataset e os tipos de dados apresentados pelo dataset

View(df_treino)

#Observa se que o dataset é composto majoritariamente por variáveis numéricas,
#somente duas variáveis são caracteres: Day_of_week e WeekStatus

summary(df_treino)

#É possível observar que a variável Appliances apresenta média bastante diferente
#da mediana e valor máximo podendo indicar presença de valores outliers. 



#-----------------------ANÁLISE EXPLORATÓRIA DOS DADOS-------------------------#

#dataset apresenta valores ausentes?

sum(is.na(df_treino)) #Não!

#Vamos observar a distribuição gráficas dos atributos numéricos do dataset

df_num <- df_treino[,2:30]


df_num_long <- df_num %>%
  pivot_longer(cols = colnames(df_num))

View(df_num_long)

plot1 <- ggplot(df_num_long, aes(x = value)) + #Plota histogramas individuais
  geom_histogram() +
  facet_wrap( ~ name, scales = "free") #scale = "free" gera eixos independentes

plot1

#Os dados numéricos apresentam uma distribuição aproximadamente normal para a
#maioria das variáveis, com exceção das variáveis randômicas introduzidas no 
#dataset e de algumas outras variáveis como a appliance, lights e RH_out

#Visualizando box-plots para observar valores outliers

plot2 <- ggplot(df_num_long, aes(x = value)) + 
  geom_boxplot() +
  facet_wrap( ~ name, scales = "free")

plot2

#Em vista que a variável preditora apresenta muitos valores outliers, iremos
#filtrar o consumo para < 150 KW

df_treino <- df_treino %>% filter(Appliances < 150)


#Observando a correlação entre as variáveis numéricas
#Obs: a correlação analisa apenas correlação entre variáveis numéricas

cor_df <- cor(df_num)
cor_df

#Podemos perceber que o consumo (variável Appliances) apresenta baixos 
#níveis de correlação com as variáveis light, T2, T6, T_out, RH_out e NSM.
#As demais variáveis apresentam correlações abaixo de 0.10/-0.10

#visualizando a matriz de correlação de forma gráfica para facilitar

corrplot(cor_df, method = 'color')



#Primeiro vamos analisar o consumo baseado nos dias da semana

df_treino$Day_of_week <- factor(df_treino$Day_of_week,levels = c(
  'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
))

consumo_por_dia <- df_treino %>% 
  group_by(Day_of_week) %>% 
  summarise(consumo = mean(Appliances))

consumo_por_dia

plot3 <- ggplot(consumo_por_dia, aes(x=Day_of_week, y=consumo)) + 
  geom_bar(stat = "identity")

plot3


#Observamos que os dias de sexta a segunda-feira temos um maior consumo 
#de energia

#Consumo baseado em horas

#Função para extrair informações de datas -> 
#as.POSIXct(dates, format = "%m/%d/%Y %H:%M:%S")

#ficam como string
# df_treino$horario <- format(as.POSIXct(df_treino$date), format = "%H:%M")
# df_treino$mes <- format(as.POSIXct(df_treino$date), format = "%m")

df_treino$horario <- hour(df_treino$date)
df_treino$mes <- month(df_treino$date)

plot4 <- ggplot(df_treino, aes(x=horario, y=mean(Appliances))) + 
  geom_bar(stat = "identity")

plot4

#---------------------------ESCALONANDO VARIÁVEIS------------------------------#

#                            VARIÁVEIS NUMÉRICAS

#obs: Irei salvar a média e desvio padrão para retornar a variável ao valor normal

media <- mean(df_treino[,2])
desvio <- sd(df_treino[,2])

df_treino_esc <- scale(df_treino[,2:30], center = TRUE, scale = TRUE)
df_treino_esc <- as.data.frame(df_treino_esc)

#usando CARET
?preProcess
pipeline <- preProcess(df_treino[,2:30], method = c("center", "scale"))
df_treino_esc<- predict(pipeline, df_treino[,2:30])
df_teste_esc <- predict(pipeline, df_teste[,2:30])


#Variáveis categóricas

labels1 <- LabelEncoder.fit(df_treino$WeekStatus)
df_treino_esc$WeekStatus <- transform(labels1, df_treino$WeekStatus)

labels2 <- LabelEncoder.fit(df_treino$Day_of_week)
df_treino_esc$Day_of_week <- transform(labels2, df_treino$Day_of_week)


#unscaled <- scaled*sd + m



#---------------------------ATRIBUTOS IMPORTANTES------------------------------#

random_forest <- randomForest(Appliances ~ . ,data = df_treino_esc, ntree = 500,
                              importance= TRUE)

summary(random_forest)
importance(random_forest)

#------------------------------------------------------------------------------#


saveRDS(random_forest, "random_forest_iot.rds")

modelo_v1 <- readRDS("random_forest_iot.rds")

#------------------------------------------------------------------------------#
#                     PREPARANDO OS DADOS DE TESTE

df_teste_esc <- scale(df_teste[,2:30], center = TRUE, scale = TRUE)
df_teste_esc <- as.data.frame(df_teste_esc)
labels3 <- LabelEncoder.fit(df_teste$WeekStatus)
df_teste_esc$WeekStatus <- transform(labels3, df_teste$WeekStatus)
labels4 <- LabelEncoder.fit(df_teste$Day_of_week)
df_teste_esc$Day_of_week <- transform(labels4, df_teste$Day_of_week)


#------------------------------------------------------------------------------#
#                   REALIZANDO PREVISÕES COM MODELO BASE

previsoes <- predict(modelo_v1, df_teste_esc[-1])

r_mse <- rmse(df_teste_esc[,1], previsoes)
mean_squared_error <- mse(df_teste_esc[,1], previsoes)

r_mse 
mean_squared_error
#------------------------------------------------------------------------------#
#               TREINANDO UM MODELO DE GRADIENTE BOOSTING


gradient_boosting <- h2o.gbm()
check_model

#parei aqui


           