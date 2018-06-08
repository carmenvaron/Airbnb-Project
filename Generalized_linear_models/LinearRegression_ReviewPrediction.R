## -------------------------------------------------------------------------
## SCRIPT: Estimación de las Reviews en el 2017 con modelos lineales
## -------------------------------------------------------------------------

## -------------------------------------------------------------------------

##### 1. Bloque de inicializacion de librerias #####

if (!require("e1071")){
  install.packages("e1071")
  library("e1071")
}

if (!require("ROCR")){
  install.packages("ROCR")
  library("ROCR")
}

if (!require("glmnet")){
  install.packages("glmnet") 
  library("glmnet")
}

if (!require("caTools")){
  install.packages("caTools") 
  library(caTools)
}

setwd("~/Máster Data Science/Proyecto Airbnb")

## -------------------------------------------------------------------------
##       PARTE 1: Métodos de Regularización
## -------------------------------------------------------------------------

## -------------------------------------------------------------------------

##### 2. Bloque de carga de datos #####

Listings<-read.csv(file="listingsHome3.csv", header = TRUE, sep = ",", dec = ".", 
                   colClasses = c(rep("numeric", 9),rep("character",3), rep("factor", 4)))

## -------------------------------------------------------------------------

##### 3. Bloque de revisión basica del dataset #####

str(Listings)
head(Listings)
tail(Listings)
summary(Listings)

## -------------------------------------------------------------------------

##### 4. Bloque de análisis gráfico #####

hist(Listings$price)
hist(log(Listings$price))
boxplot(Listings$price)

hist(Listings$X2017_Total_Number_Reviews)
hist(log(Listings$X2017_Total_Number_Reviews))
boxplot(Listings$X2017_Total_Number_Reviews)


## -------------------------------------------------------------------------

##### 5. Bloque de creación de conjuntos de entrenamiento y test #####

set.seed(12345) 
SAMPLE = sample.split(Listings$X2017_Total_Number_Reviews, SplitRatio = 0.75)
Train = subset(Listings, SAMPLE == TRUE)
Test = subset(Listings, SAMPLE == FALSE)

## -------------------------------------------------------------------------

##### 6. Bloque de parámetros básicos Regularización #####

#variables=c("latitude","longitude","accommodates", "bathrooms", "bedrooms", "minimum_nights", "review_scores_value", "price", "property_type", "cancellation_policy", "neighbourhood_cleansed", "instant_bookable", "is_business_travel_ready", "Parking", "wifi")
variables=c("latitude","longitude","accommodates", "bathrooms", "bedrooms", "review_scores_value", "price", "property_type", "cancellation_policy", "instant_bookable", "is_business_travel_ready", "Wifi")
variables=c("latitude","longitude","accommodates", "bathrooms", "bedrooms", "review_scores_value", "price")
modelo = lm(X2017_Total_Number_Reviews ~ ., data=Train[c("X2017_Total_Number_Reviews",variables)])
summary(modelo)

Lambda=50000
Pruebas=200

## -------------------------------------------------------------------------

##### 7. Bloque de regression Ridge #####

alphaSeleccionado=0

coeficientes=matrix(0,nrow=Pruebas,ncol=length(variables)+1)
coeficientes=as.data.frame(coeficientes)
colnames(coeficientes)=c("termino_independiente",variables)

metricas=data.frame(sceTrain=rep(0,Pruebas),sceTest=rep(0,Pruebas))

for (i in 1:Pruebas){
  modelo_glmnet=glmnet(x=as.matrix(Train[,variables]),y=Train$X2017_Total_Number_Reviews,lambda=Lambda*(i-1)/Pruebas,alpha=alphaSeleccionado)
  coeficientes[i,]=c(modelo_glmnet$a0,as.vector(modelo_glmnet$beta))
  
  prediccionesTrain=predict(modelo_glmnet,newx = as.matrix(Train[,variables]))
  metricas$sceTrain[i]=sum((Train$X2017_Total_Number_Reviews-prediccionesTrain)^2)
  
  prediccionesTest=predict(modelo_glmnet,newx = as.matrix(Test[,variables]))
  metricas$sceTest[i]=sum((Test$X2017_Total_Number_Reviews-prediccionesTest)^2)
}

# Gráfico con evolución de los coeficientes
colores=rainbow(length(variables))
plot(coeficientes[,1],type="l",col="blue",ylim=c(-0.5,0.5))
for (i in 1:length(variables)){
  lines(coeficientes[,i+1],type="l",col=colores[i])
}

# Gráfico con evolución de los errores
par(mar = c(5,5,2,5)) #cambiamos la configuración de la parte gráfica
plot(metricas$sceTrain,col="red",type="l",ylab="Error Train",xlab="Prueba")
par(new = T)
plot(metricas$sceTest,col="blue",type="l",axes=FALSE,xlab=NA, ylab=NA)
axis(side = 4)
mtext(side = 4, line = 3, 'Error Test')
par(mar = c(5.1,4.1,4.1,2.1)) #volvemos a la configuración inicial

# Selección del Lambda Óptimo
min(metricas$sceTest)
which(metricas$sceTest==min(metricas$sceTest))
Caso=3

# Modelo y Parámetro
metricasRidge=metricas[Caso,]
lambdaRidge=Lambda*(Caso-1)/Pruebas
coeficientesRidge=coeficientes[Caso,]
modeloRidge=glmnet(x=as.matrix(Train[,variables]),y=Train$X2017_Total_Number_Reviews,lambda=Lambda*(Caso-1)/Pruebas,alpha=alphaSeleccionado)
modeloRidge$beta

## -------------------------------------------------------------------------

##### 8. Bloque de regression Lasso #####

alphaSeleccionado=1

coeficientes=matrix(0,nrow=Pruebas,ncol=length(variables)+1)
coeficientes=as.data.frame(coeficientes)
colnames(coeficientes)=c("termino_independiente",variables)

metricas=data.frame(sceTrain=rep(0,Pruebas),sceTest=rep(0,Pruebas))

for (i in 1:Pruebas){
  modelo_glmnet=glmnet(x=as.matrix(Train[,variables]),y=Train$X2017_Total_Number_Reviews,lambda=Lambda*(i-1)/Pruebas,alpha=alphaSeleccionado)
  coeficientes[i,]=c(modelo_glmnet$a0,as.vector(modelo_glmnet$beta))
  
  prediccionesTrain=predict(modelo_glmnet,newx = as.matrix(Train[,variables]))
  metricas$sceTrain[i]=sum((Train$X2017_Total_Number_Reviews-prediccionesTrain)^2)
  
  prediccionesTest=predict(modelo_glmnet,newx = as.matrix(Test[,variables]))
  metricas$sceTest[i]=sum((Test$X2017_Total_Number_Reviews-prediccionesTest)^2)
}

# Gráfico con evolución de los coeficientes
colores=rainbow(length(variables))
plot(coeficientes[,1],type="l",col="white",ylim=c(-0.1,0.1))
for (i in 1:length(variables)){
  lines(coeficientes[,i+1],type="l",col=colores[i])
}

# Gráfico con evolución de los errores
par(mar = c(5,5,2,5)) #cambiamos la configuración de la parte gráfica
plot(metricas$sceTrain,col="red",type="l",ylab="Error Train",xlab="Prueba")
par(new = T)
plot(metricas$sceTest,col="blue",type="l",axes=FALSE,xlab=NA, ylab=NA)
axis(side = 4)
mtext(side = 4, line = 3, 'Error Test')
par(mar = c(5.1,4.1,4.1,2.1)) #volvemos a la configuración inicial

# Selección del Lambda Óptimo
min(metricas$sceTest)
which(metricas$sceTest==min(metricas$sceTest))
Caso=1001 #Demasiados casos...

# Modelo y Parámetro
metricasLasso=metricas[Caso,]
lambdaLasso=Lambda*(Caso-1)/Pruebas
coeficientesLasso=coeficientes[Caso,]
modeloLasso=glmnet(x=as.matrix(Train[,variables]),y=Train$X2017_Total_Number_Reviews,lambda=Lambda*(Caso-1)/Pruebas,alpha=alphaSeleccionado)
modeloLasso$beta

## -------------------------------------------------------------------------

##### 9. Bloque de regression Elastic Net #####

alphaSeleccionado=0.5

coeficientes=matrix(0,nrow=Pruebas,ncol=length(variables)+1)
coeficientes=as.data.frame(coeficientes)
colnames(coeficientes)=c("termino_independiente",variables)

metricas=data.frame(sceTrain=rep(0,Pruebas),sceTest=rep(0,Pruebas))

for (i in 1:Pruebas){
  modelo_glmnet=glmnet(x=as.matrix(Train[,variables]),y=Train$X2017_Total_Number_Reviews,lambda=Lambda*(i-1)/Pruebas,alpha=alphaSeleccionado)
  coeficientes[i,]=c(modelo_glmnet$a0,as.vector(modelo_glmnet$beta))
  
  prediccionesTrain=predict(modelo_glmnet,newx = as.matrix(Train[,variables]))
  metricas$sceTrain[i]=sum((Train$X2017_Total_Number_Reviews-prediccionesTrain)^2)
  
  prediccionesTest=predict(modelo_glmnet,newx = as.matrix(Test[,variables]))
  metricas$sceTest[i]=sum((Test$X2017_Total_Number_Reviews-prediccionesTest)^2)
}

# Gráfico con evolución de los coeficientes
colores=rainbow(length(variables))
plot(coeficientes[,1],type="l",col="blue",ylim=c(-0.5,0.5))
for (i in 1:length(variables)){
  lines(coeficientes[,i+1],type="l",col=colores[i])
}

# Gráfico con evolución de los errores
par(mar = c(5,5,2,5)) #cambiamos la configuración de la parte gráfica
plot(metricas$sceTrain,col="red",type="l",ylab="Error Train",xlab="Prueba")
par(new = T)
plot(metricas$sceTest,col="blue",type="l",axes=FALSE,xlab=NA, ylab=NA)
axis(side = 4)
mtext(side = 4, line = 3, 'Error Test')
par(mar = c(5.1,4.1,4.1,2.1)) #volvemos a la configuración inicial

# Selección del Lambda Óptimo
min(metricas$sceTest)
which(metricas$sceTest==min(metricas$sceTest))
Caso=200

# Modelo y Parámetro
metricasElasticNet=metricas[Caso,]
lambdaElasticNet=Lambda*(Caso-1)/Pruebas
coeficientesElasticNet=coeficientes[Caso,]
modeloElasticNet=glmnet(x=as.matrix(Train[,variables]),y=Train$X2017_Total_Number_Reviews,lambda=Lambda*(Caso-1)/Pruebas,alpha=alphaSeleccionado)
modeloElasticNet$beta

## -------------------------------------------------------------------------

##### 10. Bloque de comparativa de Modelos #####

metricasRidge
metricasLasso
metricasElasticNet
