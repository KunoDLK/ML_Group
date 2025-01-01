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

# ===============================================================================


# Load libraries
library(keras)
library(deepnet)
library(caret)

# ================================ Format The Data ==============================


# Select only necessary columns
features <- c("latitude", "longitude", "neighbourhood_group", "neighbourhood",
              "room_type")

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

training_set$latitude <- training_set$latitude * 1000
training_set$longitude <- training_set$longitude * 1000

testing_set$latitude <- testing_set$latitude * 1000
testing_set$longitude <- testing_set$longitude * 1000

print(dim(training_set))
print(str(training_set))
print(dim(testing_set))
print(str(testing_set))
# =================================================================================================

# ================================ One-Hot Encoding ================================================


# =================================================================================================

# ================================ Define the Model ===============================================
library(keras)

# Define the neural network model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = ncol(training_set) - 1) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'linear')

# =================================================================================================

# ================================ Compile the Model ==============================================
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mean_absolute_error')
)

# =================================================================================================

# ================================ Train the Model ================================================
history <- model %>% fit(
  x = as.matrix(training_set[, -ncol(training_set)]),
  y = training_set$price,
  epochs = 50, 
  batch_size = 32, 
  validation_split = 0.2
)

# =================================================================================================

# ================================ Evaluate the Model =============================================
model %>% evaluate(as.matrix(testing_set[, -ncol(testing_set)]), testing_set$price)

# =================================================================================================

# ================================ Visualize Training History =====================================
library(ggplot2)

plot(history) +
  ggtitle("Model Training History") +
  theme_minimal()

# =================================================================================================