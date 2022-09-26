#PROJETO DE PREVISÃO DE USO ENERGÉTICO RESIDENCIAL (IoT)

#Projeto da Data Science Academy no qual há um feedback para avaliação de aprendizado


#Setando o diretório do projeto
setwd("machine_learning/Datasets e projetos/Projetos-7-8/Projeto 8")
getwd()

#carregando os pacotes a serem utilizados

library("dplyr")
library("tidyr")
library('ggplot2')
library('corrplot')

#para descarrecar o pacote:
'detach(package:tidyr, unload=TRUE)'


#Carregando os dados

df_treino <- read.csv("projeto8-data_files/projeto8-training.csv") 
df_teste <- read.csv("projeto8-data_files/projeto8-testing.csv")

#Verificando o dataset e os tipos de dados apresentados pelo dataset

View(df_treino)
str(df_treino)

'Observa se que o dataset é composto majoritariamente por variáveis numéricas,
somente duas variáveis são caracteres: Day_of_week e WeekStatus'


#Análise exploratória de dados

cor_df <- cor(df_treino[,c(2:30)])
corrplot(cor_df, method = 'color')


df_treino$Day_of_week <- factor(df_treino$Day_of_week,levels = c(
  'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
))

consumo_por_dia <- df_treino %>% group_by(Day_of_week, ) %>% 
  summarise(consumo = mean(Appliances))

consumo_por_dia

ggplot(consumo_por_dia, aes(x=as.factor(Day_of_week), y=consumo)) + 
  geom_bar(stat = "identity", color='black', fill='blue')

df_treino[,c(2:30)]



           