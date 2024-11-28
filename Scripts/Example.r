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

install.packages("caret")

# ===============================================================================


# Load libraries
library(caret)

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
