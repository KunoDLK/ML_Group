# ========================= Import the data from CSV=============================

# Import CSV file
AirbnbData <- read.csv("DataSet/AB_NYC_2019.csv")

str(AirbnbData)

# Load necessary library
library(dplyr)

# Determine the most popular room type
most_popular_room_type <- AirbnbData %>%
  group_by(room_type) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  slice(1) %>%
  pull(room_type)

# Filter the data by the most popular room type
AirbnbData <- AirbnbData %>%
  filter(room_type == most_popular_room_type)

# Display the structure of the filtered data
str(AirbnbData)

# ===============================================================================

# ================== This is for running on whole dataset =======================

# Create Train and Test subsets
set.seed(25)
train_indices <- sample(seq_len(nrow(AirbnbData)), size = 0.7 * nrow(AirbnbData))

training_full <- AirbnbData[train_indices, ]
testing_full <- AirbnbData[-train_indices, ]

# ===============================================================================

# =========== This will create a smaller set for faster testing of code =========

# Create Train and Test subsets
set.seed(25)

training_small <- AirbnbData[sample(seq_len(nrow(AirbnbData)), size = 700), ]
testing_small <- AirbnbData[sample(seq_len(nrow(AirbnbData)), size = 300), ]

# ===============================================================================

# ============================ Set the datasets to use  =========================

# uncomment for small
#training <- training_small
#testing <- testing_small

# uncomment for full
training <- training_full
testing <- testing_full

# ===============================================================================

# ======================== Install Any Required Packages ========================

#install.packages("caret")

# ===============================================================================


# Load libraries
library(keras)
library(deepnet)
library(caret)

# ================================ Format The Data ==============================


# Select only necessary columns
features <- c("latitude", "longitude")

training_set <- training[, c(features, "price")]
testing_set <- testing[, c(features, "price")]

# Check and handle missing values
training_set <- na.omit(training_set)
testing_set <- na.omit(testing_set)

# Verify the dimensions and structure after cleanup

print(dim(training_set))
print(str(training_set))
print(dim(testing_set))
print(str(testing_set))

# ===============================================================================


# ================================ Normalize Latitude and Longitude ==============================

# Offset by minimum and normalize using range.
min_latitude <- min(training_set$latitude)
min_longitude <- min(training_set$longitude)

training_set$latitude <- training_set$latitude - min_latitude
training_set$longitude <- training_set$longitude - min_longitude

testing_set$latitude <- testing_set$latitude - min_latitude
testing_set$longitude <- testing_set$longitude - min_longitude

# Normalize to a range of [0, 1]
lat_range <- max(training_set$latitude) - min(training_set$latitude)
long_range <- max(training_set$longitude) - min(training_set$longitude)

training_set$latitude <- training_set$latitude / lat_range
training_set$longitude <- training_set$longitude / long_range

testing_set$latitude <- testing_set$latitude / lat_range
testing_set$longitude <- testing_set$longitude / long_range


print(dim(training_set))
print(str(training_set))
print(dim(testing_set))
print(str(testing_set))
# =================================================================================================

# ================================ Encoding ================================================

library(keras)


# Normalizing numerical variables
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

training_set$latitude <- normalize(training_set$latitude)
training_set$longitude <- normalize(training_set$longitude)
testing_set$latitude <- normalize(testing_set$latitude)
testing_set$longitude <- normalize(testing_set$longitude)

# Normalize target variable with the same method
price_min <- min(training_set$price)
price_max <- max(training_set$price)

# Normalize the price based on the training set min/max
training_set$price <- (training_set$price - price_min) / (price_max - price_min)
testing_set$price <- (testing_set$price - price_min) / (price_max - price_min)

print(dim(training_set))
print(str(training_set))
print(dim(testing_set))
print(str(testing_set))
# =================================================================================================

# ================================ Define the Model ===============================================

# Load necessary libraries
library(keras)

# Define the neural network model
model <- keras_model_sequential() %>%
  # Add layers to the model
  layer_dense(units = 32, activation = 'relu', input_shape = c(2)) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 1)

# Compile the model
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(
    learning_rate = 0.00001  # specify learning rate
  ),
  metrics = c('mean_absolute_error')
)

# Fit the model to the training data
history <- model %>% fit(
  as.matrix(training_set[, c("latitude", "longitude")]), # input features
  training_set$price,                                  # target variable
  epochs = 50,
  batch_size = 32,
  validation_split = 0.1
)

# Evaluate the model's performance on the testing data
evaluation <- model %>% evaluate(
  as.matrix(testing_set[, c("latitude", "longitude")]),
  testing_set$price
)

# Print the evaluation result
print(evaluation)

# You can predict the testing set using the trained model
predictions <- model %>% predict(as.matrix(testing_set[, c("latitude", "longitude")]))

# Load necessary library for plotting
library(ggplot2)

# Predictions need to be denormalized to match original price scale
denormalized_predictions <- predictions * (price_max - price_min) + price_min
denormalized_test_values <- testing_set$price * (price_max - price_min) + price_min

# Calculate the absolute errors
absolute_errors <- abs(denormalized_predictions - denormalized_test_values)

head(denormalized_predictions)
head(denormalized_test_values)
head(absolute_errors)

# Calculate minimum, maximum, and average absolute error
min_error <- min(absolute_errors)
max_error <- max(absolute_errors)
avg_error <- median(absolute_errors)

# Print the error statistics
cat("Minimum Absolute Error:", min_error, "\n")
cat("Maximum Absolute Error:", max_error, "\n")
cat("Median Absolute Error:", avg_error, "\n")

# Create a data frame for plotting
error_data <- data.frame(Actual_Price = denormalized_test_values,
                         Predictions = denormalized_predictions, 
                         Absolute_Error = absolute_errors)

# Plot the absolute errors
ggplot(error_data, aes(x = seq_along(Absolute_Error), y = Absolute_Error)) +
  geom_line(color = "blue") +
  labs(title = "Test Set Absolute Errors",
       x = "Test Sample Index",
       y = "Absolute Error") +
  theme_minimal()

