# notes

#### for loading and handle the new dataset :

list variable in workplace :
- ls()

~~~
# define the filename

filename <- "iris.csv"

# load the CSV file from the local directory

dataset <- read.csv(filename, header=FALSE)

# set the column names in the dataset

colnames(dataset) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")

~~~