
library(lattice)
library(ggplot2)
library(caret) # ml
library(mice) ## missing data
library(dplyr) # data manipulation
library(tidyr)
library(stringr)
library(readr)
library(splitstackshape)
library(reshape)

## LOAD CSV

data <- read_csv("X_train.csv")
#data <- subset(data, select = -c(id))
data_predict <- read_csv("X_test.csv")
#data_predict <- subset(data_predict, select = -c(id))
submit <- read_csv("y_train.csv")

## handle data

if(length(problems(data)$row) > 0){
  lignes_problematiques <- unique(problems(data)$row)
  data <- data[-lignes_problematiques,] 
}

data <- inner_join(data, submit, by ='id')

dataset <- bind_rows(data, data_predict)

##

str(dataset)
names(dataset)
head(dataset)
dim(dataset)


##

dataset$taille <- as.numeric(dataset$taille)

dataset$poids[is.na(dataset$poids)] <- -999
dataset$largeur_image[is.na(dataset$largeur_image)] <- -999
dataset$longueur_image[is.na(dataset$longueur_image)] <- -999

#Impute les valeurs manquantes de la variable catégorie par la modalité la plus fréquente).
dataset$categorie[is.na(dataset$categorie)] <- 'mode'

#Change le type de la variable "categorie" en factor.
dataset$categorie <- factor(dataset$categorie)

dataset$nom_magasin <- as.factor(dataset$nom_magasin)

dataset <- subset(dataset, select = c("id", "categorie", "poids", "prix", "nb_images", "largeur_image", "longueur_image", "nom_magasin" ,"delai_vente"))

sapply(dataset, function(x) sum(is.na(x)))
##

## train

data <- dataset[1:nrow(data), ]
data_predict <- dataset[(nrow(data)+1):nrow(dataset), ]

##

#categorie + poids + prix + nb_images + largeur_image + longueur_image


data$delai_vente <- as.factor(data$delai_vente)
sapply(data, function(x) sum(is.na(x)))


preProcess=c("center", "scale")
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"

# Stochastic Gradient Boosting (Generalized Boosted Modeling)
set.seed(seed)
fit.gbm <- train(delai_vente~., data=data, method="gbm", metric=metric, trControl=control, verbose=FALSE)

# result
results <- resamples(list(lda=fit.lda, glmnet=fit.glmnet,
                          knn=fit.knn, c50=fit.c50,
                          bagging=fit.treebag, rf=fit.rf, gbm=fit.gbm))
# Table comparison
summary(results)
# compare accuracy of models
dotplot(results)

fit.gbm.predict <- predict(fit.gbm, data_predict, type = "prob")

table(fit.gbm.predict, data_predict$delai_vente)

#86.84% accuracy
#(sum(fit.c50.predict == submit$Survived)) / nrow(submit)

solution <- data.frame(id = data_predict$id , fit.gbm.predict) 


# Write the solution to file
write.table(solution, file= 'my_prediction.csv', sep= ',', row.names = F, quote = F)

#####################################################3

K <- 5 # on partitionne l'echantillon de train en 5
set.seed(123) # 
train$cv_id <- sample(1:K, nrow(train), replace = TRUE)

logloss_vector <- c()

for(i in 1:K){
  train_cv <- train[train$cv_id != i, ]
  test_cv  <- train[train$cv_id == i, ]
  
  rf <- randomForest(data = train_cv,
                     as.factor(delai_vente) ~ categorie + poids + prix + nb_images + largeur_image + longueur_image, ntree = 10)
  
  pred <- predict(rf, test_cv, type = "prob")
  logloss <- MultiLogLoss(y_true = test_cv$delai_vente, 
                          y_pred = pred)
  
  
  print(logloss)
  logloss_vector <- append(logloss_vector, logloss)
  
}
print(paste0('Moyenne score CV : ', mean(logloss_vector)))

##################################################3