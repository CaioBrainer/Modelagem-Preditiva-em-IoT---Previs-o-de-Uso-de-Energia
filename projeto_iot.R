#PROJETO DE PREVISÃO DE USO ENERGÉTICO RESIDENCIAL (IoT)

#Projeto da Data Science Academy no qual há um feedback para avaliação de aprendizado


#Setando o diretório do projeto
#setwd("machine_learning/Datasets e projetos/Projetos-7-8/Projeto 8/Modelagem_Preditiva_em_IoT/")
getwd()

#carregando os pacotes a serem utilizados

library("dplyr")
library("tidyr")
library('ggplot2')
library('corrplot')
library('lubridate')

#para descarrecar o pacote:
'detach(package:tidyr, unload=TRUE)'


#Carregando os dados

df_treino <- read.csv("projeto8-data_files/projeto8-training.csv") 
df_teste <- read.csv("projeto8-data_files/projeto8-testing.csv")


#Verificando o dataset e os tipos de dados apresentados pelo dataset

View(df_treino)
View(df_teste)

#Observa se que o dataset é composto majoritariamente por variáveis numéricas,
#somente duas variáveis são caracteres: Day_of_week e WeekStatus

summary(df_treino)

#----------------------ANÁLISE EXPLORATÓRIA DOS DADOS-------------------------#

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
  geom_bar(stat = "identity", color='black')

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

plot4 <- ggplot(df_treino, aes(x=horario2, y=mean(Appliances))) + 
  geom_bar(stat = "identity", color='black')

plot4


#parei aqui




           