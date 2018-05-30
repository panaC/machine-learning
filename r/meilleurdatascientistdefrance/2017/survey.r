## usefull function
sapply(trsf, function(x) sum(x))

library(splitstackshape)
cSplit(dataset, "voies.admin", ',')

tmp_2 <- str_split_fixed(dataset$voies.admin, ',', Inf)

#####
## start
#####

library(lattice)
library(ggplot2)
library(caret) # ml
library(mice) ## missing data
library(dplyr) # data manipulation
library(tidyr)
library(stringr)

## LOAD CSV

data <- read.csv("boites_medicaments_train.csv", sep = ';')
data_predict <- read.csv("boites_medicaments_test.csv", sep = ';')

## handle data

dataset <- bind_rows(data, data_predict)

##

str(dataset)
names(dataset)
head(dataset)
dim(dataset)

##

### rm col unused

# backup
dataset_origin <- dataset

dataset <- subset(dataset, select = -c(libelle))
# apply factor for each libelle col
dataset[grep("libelle", names(dataset))] <- lapply(dataset[grep("libelle", names(dataset))], function(x) as.factor(x))
str(dataset)
# rm uniqu level factor
dataset$etat.commerc[dataset$etat.commerc == 'Déclaration de suspension de commercialisation'] <- 'Déclaration d\'arrêt de commercialisation' 
dataset$etat.commerc <- as.factor(dataset$etat.commerc)

str(dataset)

# tx.rembours as factor 5 lvl
dataset$tx.rembours <- as.factor(dataset$tx.rembours)

# voie.admin as factor 
table(dataset$statut.admin)
statut.admin.var <- c("Autorisation abrogée", "Autorisation archivée", "Autorisation retirée", "Autorisation suspendue")
dataset$statut.admin[dataset$statut.admin %in% statut.admin.var]  <- 'Autorisation retirée'
table(dataset$statut.admin)
dataset$statut.admin <- as.factor(dataset$statut.admin)
str(dataset)
table(dataset$statut.admin)

# rm titulaire and substante !! A UTILISER PLUS TARD // et ID a remettre dans le csv genere
dataset <- subset(dataset, select = -c(titulaires, substances, id))

# factor voies.admin
sort(table(dataset$voies.admin))


dataset <- dataset[]
tst <- sapply(dataset$voies.admin, function(x) strsplit(x, ',')[[1]])
tst_name <- unique(unlist(tst))

for (j in 1:length(tst_name)){
  dataset[,paste("voies.admin", tst_name[[j]], sep = '.')] <- 0
}

for (i in 1:length(tst)) {
  for (j in 1:length(tst[[i]])){
    print(tst[[i]][[j]])
    dataset[i,paste("voies.admin", tst[[i]][[j]], sep = '.')] <- 1
  }
}

# apply factor for each libelle col
dataset[grep("voies.admin.", names(dataset))] <- lapply(dataset[grep("voies.admin.", names(dataset))], function(x) as.factor(x))
dataset <- subset(dataset, select = -c(voies.admin))

tmp <- data.frame(c(1:nrow(dataset)))


tst_factor <- lapply(tst, function(x) as.factor(x))

#

# dummy variable test
dmy_type_proc <- subset(dataset, select = c(type.proc))
dmy <- dummyVars("~ .", data=dmy_type_proc)
predict(dmy, newdata = dmy_type_proc)
trsf <- data.frame(predict(dmy, newdata = dmy_type_proc))
trsf

##

str(dataset)
head(dataset)
sapply(dataset, function(x) sum(is.na(x)))


##



## train

part <- createDataPartition(data$Survived, p = 0.25, list = FALSE)
data_part <- data[part, ]
data_test <- data[-part, ]

rf_fit <- train(factor(Survived) ~ ., data = data_part, method = "rf", na.action = na.omit)

## predict

#data_test <- data_test[complete.cases(data_test),]
#data_test = na.omit(data_test)
rf_pred <- predict(rf_fit, data_test)

table(rf_pred, data_test$Survived)
#confusionMatrix(rf_pred, data_test$Survived)

## full train

preProcess=c("center", "scale")
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"

# Linear Discriminant Analysis
set.seed(seed)
fit.lda <- train(factor(Survived)~., data=data, method="lda", metric=metric, preProc=c("center", "scale"), trControl=control)
# Logistic Regression
set.seed(seed)
fit.glm <- train(factor(Survived)~., data=data, method="glm", metric=metric, trControl=control)
# GLMNET
set.seed(seed)
fit.glmnet <- train(factor(Survived)~., data=data, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
# SVM Radial
set.seed(seed)
fit.svmRadial <- train(factor(Survived)~., data=data, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
# kNN
set.seed(seed)
fit.knn <- train(factor(Survived)~., data=data, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Naive Bayes
set.seed(seed)
fit.nb <- train(factor(Survived)~., data=data, method="nb", metric=metric, trControl=control)
# CART
set.seed(seed)
fit.cart <- train(factor(Survived)~., data=data, method="rpart", metric=metric, trControl=control)
# C5.0
set.seed(seed)
fit.c50 <- train(factor(Survived)~., data=data, method="C5.0", metric=metric, trControl=control)
# Bagged CART
set.seed(seed)
fit.treebag <- train(factor(Survived)~., data=data, method="treebag", metric=metric, trControl=control)
# Random Forest
set.seed(seed)
fit.rf <- train(factor(Survived)~., data=data, method="rf", metric=metric, trControl=control)
# Stochastic Gradient Boosting (Generalized Boosted Modeling)
set.seed(seed)
fit.gbm <- train(factor(Survived)~., data=data, method="gbm", metric=metric, trControl=control, verbose=FALSE)

# result
results <- resamples(list(lda=fit.lda, logistic=fit.glm, glmnet=fit.glmnet,
                          svm=fit.svmRadial, knn=fit.knn, nb=fit.nb, cart=fit.cart, c50=fit.c50,
                          bagging=fit.treebag, rf=fit.rf, gbm=fit.gbm))
# Table comparison
summary(results)


fit.c50.predict <- predict(fit.c50, data_predict)

table(fit.c50.predict, submit$Survived)

#86.84% accuracy
(sum(fit.c50.predict == submit$Survived)) / nrow(submit)


fit.rf.predict <- predict(fit.rf, data_predict)

table(fit.rf.predict, submit$Survived)

#84.21% accuracy
(sum(fit.rf.predict == submit$Survived)) / nrow(submit)
