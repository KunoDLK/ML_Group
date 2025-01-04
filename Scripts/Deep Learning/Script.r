# ========================= Import the data from CSV=============================

# Import CSV file
AirbnbData <- read.csv("DataSet/AB_NYC_2019.csv")

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
features <- c("latitude", "longitude", "room_type", "minimum_nights", "number_of_reviews", "availability_365")

training_set <- training[, c(features, "price")]
testing_set <- testing[, c(features, "price")]

# Check and handle missing values
training_set <- na.omit(training_set)
testing_set <- na.omit(testing_set)

# Filter for items with a price under 1000
training_set <- training_set[training_set$price < 1000, ]
testing_set <- testing_set[testing_set$price < 1000, ]

# Verify the dimensions and structure after cleanup
print(dim(training_set))
print(str(training_set))
print(dim(testing_set))
print(str(testing_set))

# ===============================================================================

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


# ================================ Binary Encoding ================================================

# Function to convert a normalized value (0-1) to a 16-bit binary vector
normalized_to_binary_vector <- function(value) {
  # Ensure the value is within the normalized range of 0 to 1
  if (value < 0 || value > 1) {
    stop("Value must be between 0 and 1")
  }
  
  # Scale the value to a 16-bit integer
  int_value <- as.integer(value * 65535)
  
  # Convert the integer to a binary string
  binary_string <- intToBits(int_value)[1:16] # Extract the first 16 bits
  
  # Convert the binary string to a numeric vector (0s and 1s)
  binary_vector <- as.numeric(rev(binary_string)) # Reverse the bits to get the correct order
  
  return(binary_vector)
}

# Apply the binary encoding to latitude and longitude for both training and testing sets
training_lat_binary <- t(apply(as.matrix(training_set$latitude), 1, normalized_to_binary_vector))
training_long_binary <- t(apply(as.matrix(training_set$longitude), 1, normalized_to_binary_vector))

testing_lat_binary <- t(apply(as.matrix(testing_set$latitude), 1, normalized_to_binary_vector))
testing_long_binary <- t(apply(as.matrix(testing_set$longitude), 1, normalized_to_binary_vector))

# =================================================================================================

# ================================ Encoding Additional Features ===============================

# Normalize numerical variables
training_set$minimum_nights <- normalize(training_set$minimum_nights)
training_set$number_of_reviews <- normalize(training_set$number_of_reviews)
training_set$availability_365 <- normalize(training_set$availability_365)

testing_set$minimum_nights <- normalize(testing_set$minimum_nights)
testing_set$number_of_reviews <- normalize(testing_set$number_of_reviews)
testing_set$availability_365 <- normalize(testing_set$availability_365)

print(str(training_set))

# One-hot encode the room_type feature using model.matrix
# This automatically creates dummy variables for each category
training_room_type_one_hot <- model.matrix(~ room_type - 1, data = training_set)  # `-1` removes the intercept
head(training_room_type_one_hot)
testing_room_type_one_hot <- model.matrix(~ room_type - 1, data = testing_set)

# Binary encode normalized numerical features (minimum_nights, number_of_reviews, availability_365)
training_min_nights_binary <- t(apply(as.matrix(training_set$minimum_nights), 1, normalized_to_binary_vector))
head(training_min_nights_binary)
training_num_reviews_binary <- t(apply(as.matrix(training_set$number_of_reviews), 1, normalized_to_binary_vector))
head(training_num_reviews_binary)
training_avail_binary <- t(apply(as.matrix(training_set$availability_365), 1, normalized_to_binary_vector))
head(training_num_reviews_binary)
testing_min_nights_binary <- t(apply(as.matrix(testing_set$minimum_nights), 1, normalized_to_binary_vector))
testing_num_reviews_binary <- t(apply(as.matrix(testing_set$number_of_reviews), 1, normalized_to_binary_vector))
testing_avail_binary <- t(apply(as.matrix(testing_set$availability_365), 1, normalized_to_binary_vector))

# Combine all features into a single training and testing input matrix
training_inputs <- cbind(
  training_lat_binary,                 # Binary-encoded latitude
  training_long_binary,                # Binary-encoded longitude
  training_room_type_one_hot,          # One-hot encoded room type
  training_min_nights_binary,          # Binary-encoded minimum nights
  training_num_reviews_binary,         # Binary-encoded number of reviews
  training_avail_binary                # Binary-encoded availability (days per year)
)

testing_inputs <- cbind(
  testing_lat_binary,                  # Binary-encoded latitude
  testing_long_binary,                 # Binary-encoded longitude
  testing_room_type_one_hot,           # One-hot encoded room type
  testing_min_nights_binary,           # Binary-encoded minimum nights
  testing_num_reviews_binary,          # Binary-encoded number of reviews
  testing_avail_binary                 # Binary-encoded availability (days per year)
)

print(str(training_inputs))
print(str(testing_inputs))

# Update input shape in model definition to match the new input size
input_size <- ncol(training_inputs)

# Load necessary libraries
library(keras)

# Define the neural network model with updated input size
model <- keras_model_sequential() %>%
  layer_dense(units = input_size, activation = 'relu', input_shape = c(input_size)) %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 1024, activation = 'relu') %>%
  layer_dense(units = 1024, activation = 'relu') %>%
  layer_dense(units = 1024, activation = 'relu') %>%
  layer_dense(units = 1024, activation = 'relu') %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'relu')

# Compile the model
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(
    learning_rate = 0.00001 # Specify learning rate
  ),
  metrics = c('mean_absolute_error')
)

# Fit the model to the training data
history <- model %>% fit(
  training_inputs,                      # Input features (binary-encoded latitude and longitude)
  training_set$price,                   # Target variable
  epochs = 10,
  batch_size = 64,
  validation_split = 0.005
)

# Evaluate the model's performance on the testing data
evaluation <- model %>% evaluate(
  testing_inputs,                       # Input features for testing
  testing_set$price                     # Target variable for testing
)

# Print the evaluation result
print(evaluation)

# You can predict the testing set using the trained model
predictions <- model %>% predict(testing_inputs)

# Load necessary library for plotting
library(ggplot2)

# Predictions need to be denormalized to match original price scale
denormalized_predictions <- predictions * (price_max - price_min) + price_min
denormalized_test_values <- testing_set$price * (price_max - price_min) + price_min


# ================================ Output graph  ===============================================

# Load ggplot2 library
library(ggplot2)

# Assuming 'denormalized_predictions' and 'denormalized_test_values' are already calculated

# Create a data frame for plotting
plot_data <- data.frame(
  Predicted = denormalized_predictions,
  Expected = denormalized_test_values
)

# Create the scatter plot
ggplot(plot_data, aes(x = Expected, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) + # Scatter points
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") + # y=x line
  labs(
    title = "Predicted vs Expected",
    x = "Expected (Actual Prices)",
    y = "Predicted Prices"
  ) +
  theme_minimal() # Apply a clean theme

