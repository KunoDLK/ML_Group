# ========================= Import the data from CSV=============================

# Import CSV file
AirbnbData <- read.csv("DataSet/AB_NYC_2019.csv")

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
training <- training_small
testing <- testing_small

# uncomment for full
# training <- training_full
# testing <- testing_full

# ===============================================================================

# ======================== Install Any Required Packages ========================

#install.packages("caret")
#install.packages("keras")
#install.packages("ggplot2")
#library(keras)

# Install TensorFlow
#keras::install_keras()


# ===============================================================================


# Load libraries
library(caret)
library(keras)
library(ggplot2)


# ================================ Format The Data ==============================


# Select only necessary columns
features <- c("latitude", "longitude", "neighbourhood_group", "neighbourhood",
              "room_type", "minimum_nights")

training_set <- training[, c(features, "price")]
testing_set <- testing[, c(features, "price")]

# Check and handle missing values
training_set <- na.omit(training_set)
testing_set <- na.omit(testing_set)

# Ensure consistent factor levels between training and testing sets
training_set$neighbourhood <- factor(training_set$neighbourhood)
testing_set$neighbourhood <- factor(testing_set$neighbourhood, levels = levels(training_set$neighbourhood))

training_set$room_type <- factor(training_set$room_type)
testing_set$room_type <- factor(testing_set$room_type, levels = levels(training_set$room_type))

training_set$neighbourhood_group <- factor(training_set$neighbourhood_group)
testing_set$neighbourhood_group <- factor(testing_set$neighbourhood_group, levels = levels(training_set$neighbourhood_group))

# Verify the dimensions and structure after cleanup

print(dim(training_set))
print(str(training_set))
print(dim(testing_set))
print(str(testing_set))

# ===============================================================================

# ================================ Deep Learning ==============================

# Ensure that all categorical variables are numeric
dummyVars <- dummyVars(~ ., data = training_set)
training_matrix <- as.matrix(predict(dummyVars, newdata = training_set))
testing_matrix <- as.matrix(predict(dummyVars, newdata = testing_set))

# Normalize numerical features
preProcess_scale <- preProcess(training_matrix, method = c("center", "scale"))
training_matrix <- predict(preProcess_scale, newdata = training_matrix)
testing_matrix <- predict(preProcess_scale, newdata = testing_matrix)

# Define the model
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = 'relu', input_shape = ncol(training_matrix) - 1) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam(),
  metrics = c('mae')
)

# Train the model
history <- model %>% fit(
  x = training_matrix[, -ncol(training_matrix)],
  y = training_matrix[, ncol(training_matrix)],
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1
)

# Make predictions
predictions <- model %>% predict(testing_matrix[, -ncol(testing_matrix)])

# Add predictions to testing set for comparison
testing_set$predicted_price <- predictions

# Plot real vs. predicted
ggplot(testing_set, aes(x = price, y = predicted_price)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red', linetype = 'dashed') +
  theme_minimal() +
  labs(
    title = 'Actual vs Predicted Airbnb Prices',
    x = 'Actual Price',
    y = 'Predicted Price'
  )
# ===============================================================================

