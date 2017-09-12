# Importing the dataset
dataset = read.csv('50_Startups.csv')

#Encoding categorical data
dataset$State = factor(dataset$State, 
                         levels = c('New York','California','Florida'),
                         labels = c(1,2,3)
)

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)

# Fitting Multiple linear regression to training set
regressor = lm(formula = Profit ~ ., data = trainingSet)
summary(regressor)

# As we know R.D.Spend is most statistically significant from summary (least P value)
y_pred1 = predict(regressor, newdata = testSet)

regressor = lm(formula = Profit ~ R.D.Spend, data = trainingSet)
summary(regressor)

y_pred2 = predict(regressor, newdata = testSet)

# Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State , data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend , data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend , data = dataset)
summary(regressor)
#Marketting spend is on borderline - has a single dot
regressor = lm(formula = Profit ~ R.D.Spend , data = dataset)
summary(regressor)