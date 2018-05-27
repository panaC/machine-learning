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

info on data :

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

*cf -> first_iris_data.r in github*

**for missing data :**
- library(mice)
- md.pattern(data) : 0 = missing cf graph