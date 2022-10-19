#------------------------------------------------------------------------------#
#       PROJETO DE PREVISÃO DE USO ENERGÉTICO RESIDENCIAL (IoT)
#------------------------------------------------------------------------------#


#Projeto da Data Science Academy no qual há um feedback para avaliação de aprendizado


#Setando o diretório do projeto
setwd("~/machine_learning/Datasets e projetos/Projetos-7-8/Projeto 8/Modelagem_Preditiva_em_IoT/")

getwd()


#---------------------------CARREGANDO OS PACOTES------------------------------#
#carregando os pacotes a serem utilizados

library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(party)
library(Metrics)
library(CatEncoders)
library(caret)
library(performance)

#------------------------------------------------------------------------------#
#---------------------------CARREGANDO OS DADOS--------------------------------#

#Os dados fornecidos já estão divididos entre treino e teste:

df_treino <- read.csv("projeto8-data_files/projeto8-training.csv") 
df_teste <- read.csv("projeto8-data_files/projeto8-testing.csv")


#Verificando o dataset e os tipos de dados apresentados pelo dataset

View(df_treino)

#Observa se que o dataset é composto majoritariamente por variáveis numéricas,
#somente duas variáveis são caracteres: Day_of_week e WeekStatus

#A variável Appliances apresenta o consumo dos eletrodoméstricos em Wh
#As variáveis iniciadas em T, são os sensores de temperatura nos cômodos
#As variáveis iniciadas em RH, são os sensores de umidade nos cômodos
#As variáveis Tdewpoint, Press_mm_hg, Windspeed e Visibility, T_out e RH_out
#correspondem a dados climáticos na região


summary(df_treino)

#É possível observar que a variável Appliances apresenta média bastante diferente
#da mediana e valor máximo podendo indicar presença de valores outliers. 


#------------------------------------------------------------------------------#
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

#-----------------------------------------------------------------------------#
#         Visualizando box-plots para observar valores outliers

plot2 <- ggplot(df_num_long, aes(x = value)) + 
  geom_boxplot() +
  facet_wrap( ~ name, scales = "free")

plot2

#Em vista que a variável preditora apresenta muitos valores outliers, iremos
#filtrar o consumo para < 150 KW

df_treino <- df_treino %>% filter(Appliances < 150)

#-----------------------------------------------------------------------------#
#         Observando a correlação entre as variáveis numéricas

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



#Consumo baseado em horas
library(lubridate)


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

#------------------------------------------------------------------------------#
#---------------------------ESCALONANDO VARIÁVEIS------------------------------#

#                            VARIÁVEIS NUMÉRICAS


?preProcess
pipeline <- preProcess(df_treino[,3:34], method = c("center", "scale"))
df_treino_esc<- predict(pipeline, df_treino[,2:34])


#                           VARIÁVEIS CATEGÓRICAS


labels1 <- LabelEncoder.fit(df_treino$WeekStatus)
df_treino_esc$WeekStatus <- transform(labels1, df_treino$WeekStatus)

labels2 <- LabelEncoder.fit(df_treino$Day_of_week)
df_treino_esc$Day_of_week <- transform(labels2, df_treino$Day_of_week)



#unscaled <- scaled*sd + m


#------------------------------------------------------------------------------#
#---------------------------ATRIBUTOS IMPORTANTES------------------------------#

#Usaremos a biblioteca random forest para visualizar os atributos com maior
#importância para a predição de valores de consumo

library(randomForest)

random_forest <- randomForest(Appliances ~ . ,data = df_treino_esc, ntree = 500,
                              importance= TRUE)

summary(random_forest)
importance(random_forest)

#As variáveis com maiores (>40) importâncias são: lights, NSM, RH_1, RH_2, RH_5,
#T6,T7, T8, T9, T_out, Press_mm_hg, Visibility e Tdewpoint

#------------------------------------------------------------------------------#

saveRDS(random_forest, "random_forest_iot.rds")
modelo_var_imp <- readRDS("random_forest_iot.rds")
importance(modelo_var_imp)

#------------------------------------------------------------------------------#
#-------------------------PREPARANDO OS DADOS DE TESTE-------------------------#

df_teste <- df_teste %>% filter(Appliances < 150)
df_teste$horario <- hour(df_teste$date)
df_teste$mes <- month(df_teste$date)
df_teste_esc <- predict(pipeline, df_teste[,2:34])

df_teste$Day_of_week <- factor(df_teste$Day_of_week,levels = c(
  'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
))


df_teste_esc$WeekStatus <- transform(labels1, df_teste$WeekStatus)
df_teste_esc$Day_of_week <- transform(labels2, df_teste$Day_of_week)

View(df_teste_esc)

#------------------------------------------------------------------------------#
#-------------------------TREINANDO UM MODELO BASE () -------------------------#


lin_reg_v1 <- lm(Appliances ~ lights + NSM + RH_1 + RH_2 + RH_5 + T6 + T7 + T8 
                 + T9 + T_out + Press_mm_hg + Visibility + Tdewpoint,
                 data = df_treino_esc)

summary(lin_reg_v1)
check_ <- check_model(lin_reg_v1)


png(file="imagens/lin_reg_v1_metrics.png",width=1270, height=580)
check_
dev.off()

#as variáveis T6, T6, T9 e T_out apresentam alta probabilidade de colinearidade
#iremos removelas e ver o comportamento do modelo

lin_reg_v2 <- lm(Appliances ~ lights + NSM +
                   Press_mm_hg,
                 data = df_treino_esc)

#métricas de avaliação do modelo base nos dados de teste

previsões_lg <- predict(lin_reg_v1, df_teste_esc[,2:33])

rmse <- RMSE(previsões, df_teste_esc[,1]) #21.50
mse <- (RMSE(previsões, df_teste_esc[,1]))^2 #462.34
mae <- MAE(previsões, df_teste_esc[,1]) #16.28



#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
#-------------------TREINANDO UM MODELO DE GRADIENTE BOOSTING------------------#


library("gbm")

?gbm
grad_bst <- gbm(Appliances ~ lights + NSM + RH_1 + RH_2 + RH_5 + T6 + T7 + T8 
          + T9 + T_out + Press_mm_hg + Visibility + Tdewpoint,
          data = df_treino_esc, distribution = 'gaussian', n.trees = 1000,
          interaction.depth = 5)

#interation depth = 5 apresenta resultados bem melhores que o default = 1

rmse(grad_bst) #12.45
mse(grad_bst) #155.01
mae(grad_bst) #9.32

previsões_gb <- predict(grad_bst, df_teste_esc[,2:33])

rmse_gb <- RMSE(previsões_gb, df_teste_esc[,1]) #18.16
mse_gb <- (RMSE(previsões_gb, df_teste_esc[,1]))^2 #329.95
mae_gb <- MAE(previsões_gb, df_teste_esc[,1]) #13.54

saveRDS(grad_bst, "grad_bst_v1.rds")


#------------------------------------------------------------------------------#
library("h2o")


h2o.init()
dependente <- "Appliances"

independentes <- c("lights", "NSM", "RH_1", "RH_2", "RH_5", "T6", "T7", "T8", 
                   "T9", "T_out", "Press_mm_hg", "Visibility", "Tdewpoint")

df_h2o <- as.h2o(df_treino_esc)
df_h2o_teste <- as.h2o(df_teste_esc)


?h2o.gbm
grad_bst <- h2o.gbm(y=dependente, x=independentes, 
                    training_frame = df_h2o)



#avaliação
h2o.rmse(grad_bst) #15.81
h2o.mse(grad_bst) #250.15
h2o.mae(grad_bst) #11.75

performance_gbm <- h2o.performance(grad_bst, newdata = df_h2o_teste)
performance_gbm

#detach(package="h2o", unload = T)


#------------------------------------------------------------------------------#
#--------------------------TREINANDO UM MODELO DE XGboost----------------------#

library("xgboost")

?xgboost

parametros <- list(eta = 0.3) #default

X <- as.matrix(df_treino_esc[,2:33])
y <- as.matrix(df_treino_esc[,1])

xtreme_bst_v1 <- xgboost(data = X, label = y,
                         nrounds = 1000, early_stopping_rounds = 3, 
                         params = parametros, verbose = 0)



plot(xtreme_bst_v1$evaluation_log, type='l', col='blue')

X_teste <- as.matrix(df_teste_esc[,2:33])
y_teste <- as.matrix(df_treino_esc[,1])

previsões_gb <- predict(xtreme_bst_v1, X_teste)
rmse_xgb <- RMSE(previsões_gb, df_teste_esc[,1])


saveRDS(xtreme_bst_v1, "xtreme_bst_v1.rds")

#parei aqui




#------------------------------------------------------------------------------#
#                                IDEIAS

df_treino2 <- df_treino %>%
  mutate("T_media" = rowMeans(select(df_treino,c(T1,T2,T3,T4,T5
                                                 ,T6,T7,T8,T9)))) %>%
  mutate("RH_media" = rowMeans(select(df_treino,c(RH_1,RH_2,RH_3,RH_4,RH_5
                                                 ,RH_6,RH_7,RH_8,RH_9)))) %>%
  select(-c(date, T1:RH_9))


#para limpar os outliers de maneira mais efetiva!

data_clean <- data_scale %>% 
  mutate( zscore = (Appliances - mean(Appliances)) / sd(Appliances)) %>%
  filter(zscore <=3) %>%
  select(-zscore)
