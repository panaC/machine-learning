# notes

#### for loading and handle the new dataset :

list variable in workplace :
- ls()

loading data :

~~~
# define the filename

filename <- "iris.csv"

# load the CSV file from the local directory

dataset <- read.csv(filename, header=FALSE)

# set the column names in the dataset

colnames(dataset) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")

~~~

**info on data :**

- class(data)
- dim(data)
- nrow(data)
- ncol(data)
- names(data)
- head(data)
- tail(data)
- summary(data)
- table(data$col)
- str(data)

```cf -> first_iris_data.r on github```

**for missing data :**

- library(mice)
- md.pattern(data) : 0 = missing cf graph

**For known how many missing value :**

- nrow(data) - nrow(na.omit(data))

**select data.frame :**

```
iris[,1] #on sélectionne la colonne 1, c'est-à-dire la première colonne
iris[,3] #on sélectionne la colonne 3
iris[,2:3] #on sélectionne les colonnes 2 et 3
iris[,c(5,2)] #on sélectionne les colonnes 5 et 2 dans cet ordre

iris[1,] #on sélectionne la ligne 1
iris[3,] #on sélectionne la ligne 3
iris[2:3,] #on sélectionne les lignes 2 et 3
iris[c(5,2),] #on sélectionne les lignes 5 et 2 dans cet ordre
 ```
 

## Website

https://machinelearningmastery.com/evaluate-machine-learning-algorithms-with-r/
https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/