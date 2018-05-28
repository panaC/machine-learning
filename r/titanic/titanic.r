library(lattice)
library(ggplot2)
library(caret)

train <- read.csv("train.csv")
test <- read.csv("test.csv")

train <- subset(train, select = -c(Ticket, Name, Cabin))
test <- subset(test, select = -c(Ticket, Name, Cabin))

part <- createDataPartition(train$Survived, p = 0.25, list = FALSE)
data_part <- train[part, ]
data_test <- train[-part, ]

rf_fit <- train(factor(Survived) ~ ., data = train, method = "rf", na.action = na.omit)

data_test <- data_test[complete.cases(data_test),]
data_test = na.omit(data_test)
rf_pred <- predict(rf_fit, data_test, na.action = na.pass)

confusionMatrix(rf_pred, data_test$Survived)