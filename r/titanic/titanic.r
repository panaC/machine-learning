library(lattice)
library(ggplot2)
library(caret) # ml
library(mice) ## missing data
library(dplyr) # data manipulation

## LOAD CSV

data <- read.csv("train.csv")
data_predict <- read.csv("test.csv")
submit <- read.csv("gender_submission.csv")

## handle data

dataset <- bind_rows(data, data_predict)

str(dataset)
head(dataset)
sapply(dataset, function(x) sum(is.na(x)))

# extract type
dataset$Title <- gsub('(.*, )|(\\..*)', '', dataset$Name)
sapply(dataset, function(x) sum(is.na(x)))
str(dataset)
table(dataset$Title)

# compress factor level type
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
dataset$Title[dataset$Title == 'Mlle']        <- 'Miss' 
dataset$Title[dataset$Title == 'Ms']          <- 'Miss'
dataset$Title[dataset$Title == 'Mme']         <- 'Mrs'
dataset$Title[dataset$Title %in% rare_title]  <- 'Rare Title'

dataset$Title <- as.factor(dataset$Title)
dataset <- subset(dataset, select = -c(Name))

# split first letter cabin
dataset$Deck <- factor(sapply(dataset$Cabin, function(x) strsplit(x, NULL)[[1]][1]))
dataset <- subset(dataset, select = -c(Cabin))

# remove Ticket
dataset <- subset(dataset, select = -c(Ticket))

# na Fare
dataset$Fare[is.na(dataset$Fare)] <- mean(dataset$Fare, na.rm = TRUE)

# na age
dataset$Age[is.na(dataset$Age)] <- mean(dataset$Age, na.rm = TRUE)

# remove Deck many na values
dataset <- subset(dataset, select = -c(Deck))

# split data and test from dataset
data <- dataset[1:nrow(data), ]
data_predict <- dataset[(nrow(data)+1):nrow(dataset), ]

sapply(data, function(x) sum(is.na(x)))
sapply(data_predict, function(x) sum(is.na(x)))

str(data)
str(data_predict)

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
