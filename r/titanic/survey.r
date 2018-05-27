library(lattice) # visu
library(ggplot2) # visu
library(ggthemes) # visualization
library(caret) # ml
library(mice)
library(dplyr) # data manipulation
library(randomForest) # classification algorithm

train <- read.csv("train.csv", stringsAsFactors = F)
test <- read.csv("test.csv", stringsAsFactors = F)

data <- bind_rows(train, test)

str(data)

md.pattern(data)

# Grab title from passenger names
data$Title <- gsub('(.*, )|(\\..*)', '', data$Name)

# Show title counts by sex
table(data$Sex, data$Title)

# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
data$Title[data$Title == 'Mlle']        <- 'Miss' 
data$Title[data$Title == 'Ms']          <- 'Miss'
data$Title[data$Title == 'Mme']         <- 'Mrs' 
data$Title[data$Title %in% rare_title]  <- 'Rare Title'

# Show title counts by sex again
table(data$Sex, data$Title)

# Finally, grab surname from passenger name
data$Surname <- sapply(data$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])

# Create a family size variable including the passenger themselves
data$Fsize <- data$SibSp + data$Parch + 1

# Create a family variable 
data$Family <- paste(data$Surname, data$Fsize, sep='_')


## visu

# Use ggplot2 to visualize the relationship between family size & survival
ggplot(data[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat="count", position="dodge") +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = "Family Size") +
  theme_few()

# Discretize family size
data$FsizeD[data$Fsize == 1] <- "singleton"
data$FsizeD[data$Fsize < 5 & data$Fsize > 1] <- "small"
data$FsizeD[data$Fsize > 4] <- "large"

# Show family size by survival using a mosaic plot
mosaicplot(table(data$FsizeD, data$Survived), main="Family Size by Survival", shade=TRUE)

# This variable appears to have a lot of missing values
data$Cabin[1:28]

# The first character is the deck. For example:
strsplit(data$Cabin[2], NULL)[[1]]

# Create a Deck variable. Get passenger deck A - F:
data$Deck<-factor(sapply(data$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

# Get rid of our missing passenger IDs
embark_fare <- data %>%
  filter(PassengerId != 62 & PassengerId != 830)

# Use ggplot2 to visualize embarkment, passenger class, & median fare
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous()

# Since their fare was $80 for 1st class, they most likely embarked from 'C'
data$Embarked[c(62, 830)] <- 'C'

# Show row 1044
data[1044, ]

ggplot(data[data$Pclass == '3' & data$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous()

# Replace missing fare value with median fare for class/embarkment
data$Fare[1044] <- median(data[data$Pclass == '3' & data$Embarked == 'S', ]$Fare, na.rm = TRUE)

# Show number of missing Age values
sum(is.na(data$Age))

## Predictive imputation

# Make variables factors into factors
factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')

data[factor_vars] <- lapply(data[factor_vars], function(x) as.factor(x))

# Set a random seed
set.seed(129)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(data[, !names(data) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 

# Save the complete output 
mice_output <- complete(mice_mod)

# Plot age distributions
par(mfrow=c(1,2))
hist(data$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

# Replace Age variable from the mice model.
data$Age <- mice_output$Age

# Show new number of missing Age values
sum(is.na(data$Age))

## child vs mother

# First we'll look at the relationship between age & survival
ggplot(data[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  # I include Sex since we know (a priori) it's a significant predictor
  facet_grid(.~Sex)

# Create the column child, and indicate whether child or adult
data$Child[data$Age < 18] <- "Child"
data$Child[data$Age >= 18] <- "Adult"

# Show counts
table(data$Child, data$Survived)

# Adding Mother variable
data$Mother <- 'Not Mother'
data$Mother[data$Sex == 'female' & data$Parch > 0 & data$Age > 18 & data$Title != 'Miss'] <- 'Mother'

# Show counts
table(data$Mother, data$Survived)

# Finish by factorizing our two new factor variables
data$Child  <- factor(data$Child)
data$Mother <- factor(data$Mother)

md.pattern(data)

## prediction

# Split the data back into a train set and a test set
train <- data[1:891, c(1:17, 19:20)]
test <- data[892:1309,]

# split input and output
x <- train[, c(1, 3:19)]
y <- train[,2]

# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)


# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

## build model

# a) linear algorithms
set.seed(7)
fit.lda <- train(factor(Survived)~., data=train, method="lda", metric=metric, trControl=control)

# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(factor(Survived)~., data=train, method="rpart", metric=metric, trControl=control)

# kNN
set.seed(7)
fit.knn <- train(factor(Survived)~., data=train, method="knn", metric=metric, trControl=control)

# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(factor(Survived)~., data=train, method="svmRadial", metric=metric, trControl=control)

# Random Forest
set.seed(7)
fit.rf <- train(factor(Survived)~., data=train, method="rf", metric=metric, trControl=control)

# select best model

# summarize accuracy of models
results <- resamples(list(cart=fit.cart, knn=fit.knn, svm=fit.svm))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.cart)

predictions <- predict(fit.cart, train)
confusionMatrix(predictions, factor(train$Survived))

#################################3

# Set a random seed
set.seed(754)

# Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + 
                           FsizeD + Child + Mother,
                         data = train)

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

###############################3

resamples(rf_model)

# summarize accuracy of models
results <- resamples(list(cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=rf_model))
summary(results)

# compare accuracy of models
dotplot(results)

################################3

# Get importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

# Predict using the test set
prediction <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

# Write the solution to file
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)
