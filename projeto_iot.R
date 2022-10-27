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
  pipeline_mod = preProcess(df_modificado[,2:13], method = c("center", "scale"))
  df_modificado = predict(pipeline_mod, df_modificado[,1:13])
  
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
    preProcessamento <- preProcess(dataframe_iot[,3:34], method = c("center", "scale"))
    dataframe_iot <- predict(preProcessamento, dataframe_iot[,2:34])
    
    labels_Weekstatus <- LabelEncoder.fit(dataframe_iot$WeekStatus)
    dataframe_iot$WeekStatus <- transform(labels_Weekstatus, (
      dataframe_iot$WeekStatus))
    
    labels_DayofWeek <- LabelEncoder.fit(dataframe_iot$Day_of_week)
    dataframe_iot$Day_of_week <- transform(labels_DayofWeek, (
      dataframe_iot$Day_of_week))
  
    return(dataframe_iot)
} #ok


#------------------------------------------------------------------------------#
#---------------------------CARREGANDO OS DADOS--------------------------------#

#Os dados fornecidos já estão divididos entre treino e teste:

df_treino <- read.csv("projeto8-data_files/projeto8-training.csv")
#str(df_treino) #14803
df_teste <- read.csv("projeto8-data_files/projeto8-testing.csv")
#str(df_teste) #4932
df_completo <- rbind(df_treino, df_teste)

#Verificando o dataset e os tipos de dados apresentados pelo dataset

View(df_completo)

#Observa se que o dataset é composto majoritariamente por variáveis numéricas,
#somente duas variáveis são caracteres: Day_of_week e WeekStatus

#A variável Appliances apresenta o consumo dos eletrodoméstricos em Wh
#As variáveis iniciadas em T, são os sensores de temperatura nos cômodos
#As variáveis iniciadas em RH, são os sensores de umidade nos cômodos
#As variáveis Tdewpoint, Press_mm_hg, Windspeed e Visibility, T_out e RH_out
#correspondem a dados climáticos na região
#A variável NSM representam o número de segundos após meia noite


summary(df_completo)

#É possível observar que a variável Appliances apresenta média bastante diferente
#da mediana e valor máximo podendo indicar presença de valores outliers. 


#------------------------------------------------------------------------------#
#-----------------------ANÁLISE EXPLORATÓRIA DOS DADOS-------------------------#

#dataset apresenta valores ausentes?

colSums(is.na(df_completo)) #Não!


#Vamos observar a distribuição gráficas dos atributos numéricos do dataset

df_num <- df_completo[,2:30]

df_num_long <- df_num %>% 
  pivot_longer(cols = colnames(df_num))

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
boxplot(df_completo$Appliances, col='blue')
min(boxplot.stats(df_completo$Appliances)$out)

#Em vista que a variável preditora apresenta muitos valores outliers, iremos
#filtrar o consumo para < 200 KW 

df_completo <- df_completo %>% filter(Appliances < 200)

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

consumo_por_dia <- df_completo %>% 
  mutate(Day_of_week = factor(df_completo$Day_of_week,levels = c(
    'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
  ))) %>%
  group_by(Day_of_week) %>% 
  summarise(consumo = mean(Appliances))

consumo_por_dia

plot3 <- ggplot(consumo_por_dia, aes(x=Day_of_week, y=consumo)) + 
  geom_bar(stat = "identity")

plot3

#Primeiro vamos analisar o consumo baseado entre dia útil e fins de semana

consumo_por_dia_util <- df_completo %>% 
  mutate(WeekStatus = as.factor(df_completo$WeekStatus)) %>%
  group_by(WeekStatus) %>% 
  summarise(consumo = mean(Appliances))

plot4 <- ggplot(consumo_por_dia_util, aes(x=WeekStatus, y=consumo)) + 
  geom_bar(stat = "identity")

plot4

#a diferença de consumo entre dias da semana e fins de semana não aparenta ser
#considerável, são valores bem próximos


df_completo$horario <- hour(df_completo$date)
df_completo$mes <- month(df_completo$date)

plot5 <- ggplot(df_completo, aes(x=horario, y=mean(Appliances))) + 
  geom_bar(stat = "identity")

plot5
#a média de consumo é mais elevada durante as primeiras horas do dia, entre 00
#e 05 horas


#------------------------------------------------------------------------------#
#---------------------------PRÉ PROCESSANDO OS DADOS---------------------------#

#                TRATANDO VARIÁVEIS NUMÉRICAS e CATEGÓRICAS

#automatizamos o processamento através da função pipelineProcessamento
df_completo_esc <- pipelineProcessamento(df_completo)

particao <- createDataPartition(y=df_completo_esc$Appliances, p = 0.75, list=F)

treino <- df_completo_esc[particao,]
teste <- df_completo_esc[-particao,]

#------------------------------------------------------------------------------#
#---------------------------ATRIBUTOS IMPORTANTES------------------------------#

#Usaremos a biblioteca random forest para visualizar os atributos com maior
#importância para a predição de valores de consumo


random_forest_base <- randomForest(Appliances ~ . ,data = treino, ntree = 500,
                              importance= TRUE)

importance(random_forest_base)



previsoes_rf_base <- predict(random_forest_base, teste[,2:33])
avaliacao_rf_base <- avaliacaoModelo('random forest', teste$Appliances,
                                     previsoes_rf_base)

avaliacao_rf_base
saveRDS(random_forest_base, file = "random_forest_base.rds")


#As variáveis com maiores (>40) importâncias são: lights, NSM, RH_1, RH_2, RH_5,
#T6,T7, T8, T9, T_out, Press_mm_hg, Visibility e Tdewpoint


#E por que não avaliar o modelo com as variáveis mais importantes?

random_forest_v1 <- randomForest(Appliances ~ lights + T3 + T6 + T7 + T8 + T9 + T_out +
                                   RH_1 + RH_2 + RH_3 +RH_5 + Press_mm_hg + NSM, 
                                 data = treino, ntree = 500,
                                 importance= TRUE)


previsoes_rf_v1 <- predict(random_forest_v1, teste[c("lights", "T3", "T6", "T7",
                                                     "T8", "T9","T_out", "RH_1",
                                                     "RH_2", "RH_3", "RH_5",
                                                     "Press_mm_hg", "NSM")])

avaliacao_rf_v1 <- avaliacaoModelo('random forest', teste$Appliances,
                                   previsoes_rf_v1)

avaliacao_rf_v1
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#------------------TREINANDO MODELOS de REGRESSÃO LINEAR-----------------------#


lin_reg <- lm(Appliances ~ lights + T3 + T6 + T7 + T8 + T9 + T_out +
                   RH_1 + RH_2 + RH_3 +RH_5 + Press_mm_hg + NSM,
                 data = treino)


previsões_lr <- predict(lin_reg, teste[c("lights", "T3", "T6", "T7",
                                         "T8", "T9","T_out", "RH_1",
                                         "RH_2", "RH_3", "RH_5",
                                         "Press_mm_hg", "NSM")])

avaliacao_lr <- avaliacaoModelo("regressão linear", teste$Appliances,
                                 previsões_lr)

avaliacao_lr


#------------------------------------------------------------------------------#
#-------------------TREINANDO UM MODELO DE GRADIENTE BOOSTING------------------#


grad_bst <- gbm(Appliances ~ lights + NSM + RH_1 + RH_2 + RH_3 + RH_5 + T3 + T6 + T7 + T8 
          + T9 + T_out + Press_mm_hg,
          data = treino, distribution = 'gaussian', n.trees = 500,
          interaction.depth = 6, cv.folds=5)

previsões_gb <- predict(grad_bst, teste[c("lights", "T3", "T6", "T7",
                                            "T8", "T9","T_out", "RH_1",
                                            "RH_2", "RH_3", "RH_5",
                                            "Press_mm_hg", "NSM")])

avaliacao_gbm <- avaliacaoModelo('gradient boosting', teste$Appliances, previsões_gb)
avaliacao_gbm

#------------------------------------------------------------------------------#
#--------------------------TREINANDO UM MODELO DE XGboost----------------------#

?xgboost
parametros <- list(eta = 0.3, subsample = 1, max_depth=6) #default

X <- as.matrix(treino[c("lights", "T3", "T6", "T7",
                        "T8", "T9","T_out", "RH_1",
                        "RH_2", "RH_3", "RH_5",
                        "Press_mm_hg", "NSM")])

y <- as.matrix(treino[,1])

#modelo com todos as variáveis
xtreme_bst_v1 <- xgboost(data = X, label = y,
                         nrounds = 500, early_stopping_rounds = 3, 
                         params = parametros, verbose = 1)


X_teste <- as.matrix(teste[c("lights", "T3", "T6", "T7",
                             "T8", "T9","T_out", "RH_1",
                             "RH_2", "RH_3", "RH_5",
                             "Press_mm_hg", "NSM")])

y_teste <- as.matrix(teste[,1])

previsoes_xgb <- predict(xtreme_bst_v1, X_teste)

avaliacao_xgb <- avaliacaoModelo('XGBoost',as.numeric(teste$Appliances), 
                                  as.numeric(previsoes_xgb))
avaliacao_xgb



#------------------------------------------------------------------------------#
#--------------------------SALVANDO RESULTADOS---------------------------------#


resultados_metricas <- rbind(avaliacao_rf_base, avaliacao_rf_v1, avaliacao_lr,
                             avaliacao_gbm, avaliacao_xgb)
View(resultados_metricas)

write.csv(resultados_metricas, file="resultados_modelos_v1.csv")

results_v1 <- read.csv(file="resultados_modelos_v1.csv", row.names = 1)

View(results)
#------------------------------------------------------------------------------#
#                                IDEIAS


#Como alternativa, resolvi não excluir as variáveis de menor importância como
#nos primeiros modelos (e sugerido no projeto), então uni as variáveis com alta 
#colinearidade em T_media e RH_media

df_completo_mod <- modificaColunas(df_completo)

particao <- createDataPartition(y=df_completo_mod$Appliances, p = 0.75, list=F)
treino <- df_completo_mod[particao,]
teste <- df_completo_mod[-particao,]
#------------------------------------------------------------------------------#

?randomForest
rf2 <- randomForest(Appliances ~ ., data = treino, importance=T)
previsoes_rf2 <- predict(rf2, teste[,2:13])
avaliacao_rf2 <- avaliacaoModelo('random forest', teste$Appliances, previsoes_rf2)
avaliacao_rf2


#------------------------------------------------------------------------------#
X <- as.matrix(treino[,2:13])
y <- as.matrix(treino$Appliances)

parametros <- list(eta = 0.3, subsample = 0.5, max_depth=6)
xtreme_bst_v2 <- xgboost(data = X, label = y,
                         nrounds = 500, early_stopping_rounds = 3, 
                         params = parametros, verbose = 1)

xtreme_bst_v2$nfeatures


X_teste <- as.matrix(teste[,2:13])
y_teste <- as.matrix(teste$Appliances)

previsoes_xgb2 <- predict(xtreme_bst_v2, X_teste)

avaliacao_xgb2 <- avaliacaoModelo('XGBoost',as.numeric(teste$Appliances), 
                                 as.numeric(previsoes_xgb2))
avaliacao_xgb2

#------------------------------------------------------------------------------#
grad_bst2 <- gbm(Appliances ~ .,
                data = treino, distribution = 'gaussian', n.trees = 500,
                interaction.depth = 6, cv.folds=5)

#interation depth = 5 apresenta resultados bem melhores que o default = 1


previsões_gbm2 <- predict(grad_bst2, teste[,2:13])

avaliacao_gbm2 <- avaliacaoModelo('gradient boosting', teste$Appliances, previsões_gbm2)
avaliacao_gbm2

#------------------------------------------------------------------------------#
lin_reg_v2 <- lm(Appliances ~ ., data = treino)

#métricas de avaliação do modelo base nos dados de teste

previsões_lr2 <- predict(lin_reg_v2, teste[,2:13])

avaliacao_lr2 <- avaliacaoModelo("regressão linear", teste$Appliances,
                                previsões_lr2)
avaliacao_lr2


#------------------------------------------------------------------------------#
#                 COMPARANDO AS MÉTRICAS ENTRE OS MODELOS


resultados <- rbind(avaliacao_rf2, avaliacao_lr2, avaliacao_gbm2, avaliacao_xgb2)
resultados

write.csv(resultados, file="resultados_modelos_v2.csv")
results_V2 <- read.csv(file="resultados_modelos_v2.csv", row.names = 1)

results_v1$tipo_modelo <- c("variáveis mais importantes")
results_V2$tipo_modelo <- c("média das variáveis")


resultados_completo <- rbind(results_v1, results_V2)
View(resultados_completo)

ggplot(resultados_completo, aes(x=Modelo, y=RMSE, fill=tipo_modelo)) +
         geom_bar(stat="identity", position="dodge") +
         ggtitle("RMSE")

ggplot(resultados_completo, aes(x=Modelo, y=MAE, fill=tipo_modelo)) +
  geom_bar(stat="identity", position="dodge") +
  ggtitle("MAE")

ggplot(resultados_completo, aes(x=Modelo, y=R_Square, fill=tipo_modelo)) +
  geom_bar(stat="identity", position="dodge") +
  ggtitle("R_square")

#as métricas avaliadas demonstram que os modelos com variáveis selecionadas
#apresentam um desempenho ligeiramente melhor que os modelos com variáveis médias
#o modelo random forest apresenta o melhor desempenho em seus parâmetros default
#talvez sendo possível aprimorar seus resultados

#------------------------------------------------------------------------------#
#                     APRIMORANDO OS MODELOS RANDOM FOREST

#Devido ao desempenho levemente superior do modelo com todas as variáveis,
#iremos utiliza-lo em adição a função tuneRF para obter o melhor valor mtry que
#melhora os resultados do modelo.


#Utilizando tdas as variáveis
bestmtry1 <- tuneRF(treino[,2:33], treino$Appliances, ntreeTry = 500, mtryStart = 2,
                    stepFactor=1.5, improve=1e-5, doBest = T)
print(bestmtry1)
modelo_RFtuned1 <- randomForest(Appliances ~ ., data = treino, 
                                ntree=500, mtry=4, importance=TRUE)
previsao_RFtuned1 <- predict(modelo_RFtuned, teste[,2:33])
avaliacaoModelo("random forest tunning", teste$Appliances, previsao_RFtuned1) 
#rmse 11 e R²=0.85 mesmo utilizando todas as variáveis!!!

importance(modelo_RFtuned1)





