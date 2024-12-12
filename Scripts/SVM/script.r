library(e1071)
library(dplyr)  
library(caret)  
library(ggplot2)

data <- read.csv("/Users/chopelletto/Desktop/Opera Downloads/ML_Group-main/DataSet/AB_NYC_2019.csv")


data$neighbourhood <- as.factor(data$neighbourhood)


data <- data %>% select(price, neighbourhood)


data <- na.omit(data)


set.seed(25)  
trainIndex <- createDataPartition(data$price, p = .8, 
                                  list = FALSE, 
                                  times = 1)
dataTrain <- data[trainIndex, ]
dataTest <- data[-trainIndex, ]


svm_model <- svm(price ~ neighbourhood, data = dataTrain, type = 'eps-regression')


predictions <- predict(svm_model, dataTest)


mae <- mean(abs(predictions - dataTest$price))
rmse <- sqrt(mean((predictions - dataTest$price)^2))


cat("Mean Absolute Error:", mae, "\n")
cat("Root Mean Squared Error:", rmse, "\n")

ggplot(data, aes(x = price)) +
  geom_histogram(binwidth = 50, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Prices", x = "Price", y = "Frequency") +
  theme_minimal()

results <- data.frame(Actual = dataTest$price, Predicted = predictions)

ggplot(results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red") +  
  labs(title = "Actual vs Predicted Prices", x = "Actual Prices", y = "Predicted Prices") +
  theme_minimal()

results <- results %>%
  mutate(Residuals = Actual - Predicted)

ggplot(results, aes(x = Predicted, y = Residuals)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red") +  
  labs(title = "Residuals vs Predicted Prices", x = "Predicted Prices", y = "Residuals") +
  theme_minimal()
