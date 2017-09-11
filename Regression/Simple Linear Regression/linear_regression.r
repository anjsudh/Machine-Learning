# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)

regressor = lm(formula = Salary ~ YearsExperience, data = trainingSet)
summary(regressor)

# Predict test results
Y_predicted = predict(regressor, testSet)

# Visualizing the Training Set
# install.packages('ggplot2')
# library(ggplot2)
ggplot() + geom_point(aes(x = trainingSet$YearsExperience, y = trainingSet$Salary), color = 'red') + geom_line(aes(x = trainingSet$YearsExperience, y = predict(regressor, newdata = trainingSet)), color='blue') + ggtitle('Salary vs Experience - Training Set') + xlab('Years of experience') + ylab('Salary')
ggplot() + geom_point(aes(x = testSet$YearsExperience, y = testSet$Salary), color = 'red') + geom_line(aes(x = trainingSet$YearsExperience, y = predict(regressor, newdata = trainingSet)), color='blue') + ggtitle('Salary vs Experience - Test Set') + xlab('Years of experience') + ylab('Salary')
