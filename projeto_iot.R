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
library(ggplot2)
library(corrplot)
library(party)
library(Metrics)
library(CatEncoders)

library(performance)
library(psych)
library(lubridate)

library(caret)
library(randomForest)
library(gbm)
library(xgboost)

#----------------------------------FUNÇÕES-------------------------------------#
#Nesta parte estarão funções a serem utilizadas em diversos momentos 

avaliacaoModelo <- function(nome_do_modelo, observado, predito){
  
  
  data.frame(Modelo = as.character(nome_do_modelo),
             RMSE = RMSE(predito, observado),
             MAE = MAE(predito, observado),
             R_Square = R2(predito, observado),
             check.names = F
  ) %>% 
    mutate(MSE = sqrt(RMSE))
}

#pipeline para gerar um dataframe com variáveis modificadas
modificaColunas <- function(dataframe_iot) {
  
  #gerando as colunas T_media e RH_media
  df_modificado = dataframe_iot %>%
    mutate("T_media" = rowMeans(select(dataframe_iot,c(T1,T2,T3,T4,T5
                                                       ,T6,T7,T8,T9)))) %>%
    mutate("RH_media" = rowMeans(select(dataframe_iot,c(RH_1,RH_2,RH_3,RH_4,RH_5
                                                        ,RH_6,RH_7,RH_8,RH_9)))) %>%
    select(-c(date,rv1,rv2, T1:RH_9))
  
  #escalonando as variáveis numéricas
  pipeline_mod = preProcess(df_modificado[,2:11], method = c("center", "scale"))
  df_modificado = predict(pipeline_mod, df_modificado[,1:11])
  
  #tratando variáveis categóricas
  labels_Weekstatus <- LabelEncoder.fit(df_modificado$WeekStatus)
  df_modificado$WeekStatus <- transform(labels_Weekstatus, as.character(
    df_modificado$WeekStatus))
  
  labels_DayofWeek <- LabelEncoder.fit(df_treino$Day_of_week)
  df_modificado$Day_of_week <- transform(labels_DayofWeek, as.character(
    df_modificado$Day_of_week))
  
  return(df_modificado)
  
  
}

#Pipeline para processar variáveis numéricas e categoricas no dataframe completo
pipelineProcessamento <- function(dataframe_iot) {
  
  dataframe_iot = dataframe_iot
  
  
    #escalonando as variáveis numéricas
    preProcessamento <- preProcess(dataframe_iot[,3:32], method = c("center", "scale"))
    dataframe_iot <- predict(preProcessamento, dataframe_iot[,2:32])
    
    labels_Weekstatus <- LabelEncoder.fit(dataframe_iot$WeekStatus)
    dataframe_iot$WeekStatus <- transform(labels_Weekstatus, as.character(
      dataframe_iot$WeekStatus))
    
    labels_DayofWeek <- LabelEncoder.fit(dataframe_iot$Day_of_week)
    dataframe_iot$Day_of_week <- transform(labels_DayofWeek, as.character(
      dataframe_iot$Day_of_week))
  
    return(dataframe_iot)
} #ok


#------------------------------------------------------------------------------#
#---------------------------CARREGANDO OS DADOS--------------------------------#

#Os dados fornecidos já estão divididos entre treino e teste:

df_treino <- read.csv("projeto8-data_files/projeto8-training.csv")
str(df_treino) #14803

df_teste <- read.csv("projeto8-data_files/projeto8-testing.csv")
str(df_teste) #4932

df_completo <- rbind(df_treino, df_teste)

#Verificando o dataset e os tipos de dados apresentados pelo dataset

View(df_treino)

#Observa se que o dataset é composto majoritariamente por variáveis numéricas,
#somente duas variáveis são caracteres: Day_of_week e WeekStatus

#A variável Appliances apresenta o consumo dos eletrodoméstricos em Wh
#As variáveis iniciadas em T, são os sensores de temperatura nos cômodos
#As variáveis iniciadas em RH, são os sensores de umidade nos cômodos
#As variáveis Tdewpoint, Press_mm_hg, Windspeed e Visibility, T_out e RH_out
#correspondem a dados climáticos na região
#A variável NSM representam o número de segundos após meia noite


summary(df_treino)

#É possível observar que a variável Appliances apresenta média bastante diferente
#da mediana e valor máximo podendo indicar presença de valores outliers. 


#------------------------------------------------------------------------------#
#-----------------------ANÁLISE EXPLORATÓRIA DOS DADOS-------------------------#

#dataset apresenta valores ausentes?

colSums(is.na(df_treino)) #Não!


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


#-----------------------------------------------------------------------------#
boxplot(df_treino$Appliances, col='blue')
min(boxplot.stats(df_treino$Appliances)$out)

#Em vista que a variável preditora apresenta muitos valores outliers, iremos
#filtrar o consumo para < 200 KW 

df_treino <- df_treino %>% filter(Appliances < 200)

#df_treino <- df_treino %>% 
  #mutate( zscore = (Appliances - mean(Appliances)) / sd(Appliances)) %>%
  #filter(zscore <=3 & zscore >= -3) %>%
  #select(-zscore)

#-----------------------------------------------------------------------------#
#         Observando a correlação entre as variáveis numéricas


cor_df <- cor(df_num)
cor_df


#Podemos perceber que o consumo (variável Appliances) apresenta baixos 
#níveis de correlação com as variáveis light, T2, T6, T_out, RH_out e NSM.
#As demais variáveis apresentam correlações abaixo de 0.10/-0.10

#visualizando a matriz de correlação de forma gráfica para facilitar

corrplot(cor_df, method = 'color')


#-----------------------------------------------------------------------------#
#                         Gerando novas variáveis


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
preProcessamento <- preProcess(df_treino[,3:34], method = c("center", "scale"))
df_treino_esc<- predict(preProcessamento, df_treino[,2:34])


#                           VARIÁVEIS CATEGÓRICAS


labels1 <- LabelEncoder.fit(df_treino$WeekStatus)
df_treino_esc$WeekStatus <- transform(labels1, df_treino$WeekStatus)

labels2 <- LabelEncoder.fit(df_treino$Day_of_week)
df_treino_esc$Day_of_week <- transform(labels2, as.character(df_treino$Day_of_week))



#unscaled <- scaled*sd + m


#------------------------------------------------------------------------------#
#---------------------------ATRIBUTOS IMPORTANTES------------------------------#

#Usaremos a biblioteca random forest para visualizar os atributos com maior
#importância para a predição de valores de consumo


?randomForest
random_forest <- randomForest(Appliances ~ . ,data = df_treino_esc, ntree = 500,
                              importance= TRUE)


summary(random_forest)
importance(random_forest)

rmse(random_forest) #17.61
mse(random_forest) # 310.29
mean(random_forest$rsq) #0.65


#As variáveis com maiores (>40) importâncias são: lights, NSM, RH_1, RH_2, RH_5,
#T6,T7, T8, T9, T_out, Press_mm_hg, Visibility e Tdewpoint

#------------------------------------------------------------------------------#

saveRDS(random_forest, "random_forest_iot.rds")
modelo_var_imp <- readRDS("random_forest_iot.rds")
importance(modelo_var_imp)

#------------------------------------------------------------------------------#
#-------------------------PREPARANDO OS DADOS DE TESTE-------------------------#

df_teste <- df_teste %>% filter(Appliances < 200)


#df_teste <- df_teste %>% 
  #mutate( zscore = (Appliances - mean(Appliances)) / sd(Appliances)) %>%
  #filter(zscore <=3 & zscore >= -3) %>%
  #select(-zscore)


df_teste$Day_of_week <- factor(df_teste$Day_of_week,levels = c(
  'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
))


df_teste$horario <- hour(df_teste$date)
df_teste$mes <- month(df_teste$date)
df_teste_esc <- predict(pipeline, df_teste[,2:34])

df_teste_esc$WeekStatus <- transform(labels1, df_teste$WeekStatus)
df_teste_esc$Day_of_week <- transform(labels2, as.character(df_teste$Day_of_week))

View(df_teste_esc)

#------------------------------------------------------------------------------#
#------------------TREINANDO UM MODELO REGRESSÃO LINEAR -----------------------#


lin_reg_v1 <- lm(Appliances ~ lights + T3 + T6 + T7 + T8 + T9 + T_out +
                   RH_1 + RH_2 + RH_3 +RH_5 + Press_mm_hg + NSM,
                 data = df_treino_esc)


#métricas de avaliação do modelo base nos dados de teste

previsões_rin_reg <- predict(lin_reg_v1, df_teste_esc[,2:33])
summary(lin_reg_v1)

avaliacao_rl <- avaliacao_modelo("regressão linear", df_teste_esc$Appliances,
                                 previsões_rin_reg)

avaliacao_rl

saveRDS(lin_reg_v1, "lin_reg_V1.rds")

#------------------------------------------------------------------------------#
#-------------------TREINANDO UM MODELO DE GRADIENTE BOOSTING------------------#


?gbm
grad_bst <- gbm(Appliances ~ lights + NSM + RH_1 + RH_2 + RH_3 + RH_5 + T6 + T7 + T8 
          + T9 + T_out + Press_mm_hg,
          data = df_treino_esc, distribution = 'gaussian', n.trees = 500,
          interaction.depth = 10, cv.folds=5)

#interation depth = 5 apresenta resultados bem melhores que o default = 1


previsões_gb <- predict(grad_bst, df_teste_esc[,2:33])

avaliacao_gbm <- avaliacao_modelo('gradient boosting', df_teste_esc$Appliances, previsões_gb)

RMSE(previsões_gb, df_teste_esc[,1]) #18.72
sqrt(RMSE(previsões_gb, df_teste_esc[,1])) #4.32
MAE(previsões_gb, df_teste_esc[,1]) #12.63

saveRDS(grad_bst, "grad_bst_v1.rds")


#------------------------------------------------------------------------------#
#--------------------------TREINANDO UM MODELO DE XGboost----------------------#


?xgboost

parametros <- list(eta = 0.1, subsample = 0.5, max_depth=6) #default

X <- as.matrix(df_treino_esc[c("lights", "NSM", "RH_1", "RH_2", "RH_3",
                               "RH_5", "T6", "T7", "T8", "T9", "T_out", 
                               "Press_mm_hg", "Visibility", "Tdewpoint")])

y <- as.matrix(df_treino_esc[,1])

#modelo com todos as variáveis
xtreme_bst_v1 <- xgboost(data = X, label = y,
                         nrounds = 500, early_stopping_rounds = 3, 
                         params = parametros, verbose = 1)



plot(xtreme_bst_v1$evaluation_log, type='l', col='blue')

X_teste <- as.matrix(df_teste_esc[c("lights", "NSM", "RH_1", "RH_2", "RH_3", 
                                    "RH_5", "T6", "T7", "T8","T9", "T_out",
                                    "Press_mm_hg", "Visibility", "Tdewpoint")])

y_teste <- as.matrix(df_teste_esc[,1])

previsoes_xgb <- predict(xtreme_bst_v1, X_teste)

avaliacao_xgb <- avaliacao_modelo('XGBoost',as.numeric(df_teste_esc$Appliances), 
                                  as.numeric(previsoes_xgb))


saveRDS(xtreme_bst_v1, "xtreme_bst_v1.rds")

#------------------------------------------------------------------------------#
#--------------------------SALVANDO RESULTADOS---------------------------------#

?cbind
resultados_metricas <- rbind(avaliacao_rl, avaliacao_gbm, avaliacao_xgb)
View(resultados_metricas)

write.csv(resultados_metricas, file="resultados_modelos.csv")

results <- read.csv(file="resultados_modelos.csv", row.names = 1)

View(resultss)
#------------------------------------------------------------------------------#
#                                IDEIAS



