#--------------------Data Clean-------------------------------------------

# Load necessary libraries
library(tidyverse) # For data manipulation and visualisation

# Load the dataset
airbnb_data <- read.csv("AB_NYC_2019.csv")

# Inspect the data structure and basic statistics
str(airbnb_data) # Check structure of the dataset
summary(airbnb_data) # Summary statistics of the dataset
head(airbnb_data) # Preview the first few rows

# Count rows with missing or zero price before filtering is applied
rows_with_missing_or_zero_price <- airbnb_data %>%
        filter(is.na(price) | price == 0) %>%
        nrow()
cat("Rows with missing or zero price (before filtering):", rows_with_missing_or_zero_price, "\n")

# Remove rows where price is missing or 0
airbnb_data <- airbnb_data %>%
        filter(!is.na(price) & price > 0)

# Count rows with missing or zero price after filtering
rows_with_missing_or_zero_price_after <- airbnb_data %>%
        filter(is.na(price) | price == 0) %>%
        nrow()
cat("Rows with missing or zero price (after filtering):", rows_with_missing_or_zero_price_after, "\n")

# Confirm the cleaned dataset
summary(airbnb_data)

# Remove unnecessary columns (latitude and longitude)
# These are not required for modeling in the current analysis
airbnb_data <- airbnb_data %>%
        select(
                neighbourhood_group, neighbourhood, room_type, price, minimum_nights,
                number_of_reviews, reviews_per_month, calculated_host_listings_count, availability_365
        )

# Confirm changes to the dataset
glimpse(airbnb_data)

# Check for missing values in numeric columns
numeric_cols <- sapply(airbnb_data, is.numeric)
missing_counts <- colSums(is.na(airbnb_data[, numeric_cols]))

# Print columns with missing values (if any)
print(missing_counts[missing_counts > 0])

# Impute missing values in reviews_per_month with 0
airbnb_data$reviews_per_month[is.na(airbnb_data$reviews_per_month)] <- 0

# Confirm no missing values remain
colSums(is.na(airbnb_data))

# Save the cleaned dataset to a new CSV file for reproducibility
write.csv(airbnb_data, "cleaned_AB_NYC_2019.csv", row.names = FALSE)

#-------------------------------Exploratory Data Analysis-----------------------

# Reload the cleaned dataset for analysis continuity
airbnb_data <- read.csv("cleaned_AB_NYC_2019.csv")

# Inspect structure and summary
str(airbnb_data)
summary(airbnb_data)

# Histogram of price distribution
ggplot(airbnb_data, aes(x = price)) +
        geom_histogram(bins = 30, fill = "blue", color = "white") +
        theme_minimal() +
        labs(title = "Distribution of Price", x = "Price", y = "Count")

# Log-transform price to handle skewness
ggplot(airbnb_data, aes(x = log1p(price))) +
        geom_histogram(bins = 30, fill = "green", color = "white") +
        theme_minimal() +
        labs(title = "Log-Transformed Distribution of Price", x = "Log(Price + 1)", y = "Count")

# Add a new column for log-transformed price
airbnb_data$log_price <- log1p(airbnb_data$price)

# Boxplot of price by neighbourhood_group
ggplot(airbnb_data, aes(x = neighbourhood_group, y = price)) +
        geom_boxplot(fill = "orange") +
        theme_minimal() +
        labs(title = "Price by Neighbourhood Group", x = "Neighbourhood Group", y = "Price")

# Boxplot of price by room_type
ggplot(airbnb_data, aes(x = room_type, y = price)) +
        geom_boxplot(fill = "purple") +
        theme_minimal() +
        labs(title = "Price by Room Type", x = "Room Type", y = "Price")

#--------------------Prepare Data for Model-------------------------------------

# Label encode categorical variables for modeling
airbnb_data$neighbourhood_group <- as.numeric(as.factor(airbnb_data$neighbourhood_group))
airbnb_data$room_type <- as.numeric(as.factor(airbnb_data$room_type))

# Filter out extreme prices (top 1% to handle outliers)
price_threshold <- quantile(airbnb_data$price, 0.99) # 99th percentile
airbnb_data <- airbnb_data %>%
        filter(price <= price_threshold)

# Recalculate log_price after filtering
airbnb_data$log_price <- log1p(airbnb_data$price)

# Split the data into training (70%) and testing (30%) sets
set.seed(123) # For reproducibility
train_index <- sample(1:nrow(airbnb_data), size = 0.7 * nrow(airbnb_data))
train_data <- airbnb_data[train_index, ]
test_data <- airbnb_data[-train_index, ]

#----------------------Train and Evaluate the Model---------------------------------------

# Install and load necessary libraries
install.packages("Metrics")
install.packages("randomForest")
library(Metrics)
library(randomForest)

# Train the random forest model
rf_model <- randomForest(log_price ~ ., data = train_data, importance = TRUE, ntree = 100)

# Print the model summary
print(rf_model)

# Predict on the test set
test_predictions <- predict(rf_model, test_data)

# Calculate RMSE and R²
rmse_value <- rmse(test_data$log_price, test_predictions)
cat("RMSE:", rmse_value, "\n")

r2_value <- cor(test_data$log_price, test_predictions)^2
cat("R²:", r2_value, "\n")

# Feature importance visualisation
varImpPlot(rf_model)

#----------------------Visualise Predictions---------------------------------------

# Back-transform log_price predictions to actual price scale
test_actual_price <- exp(test_data$log_price) - 1
test_predicted_price <- exp(test_predictions) - 1

# Scatter plot of predicted vs. actual prices
comparison_df <- data.frame(Actual = test_actual_price, Predicted = test_predicted_price)

ggplot(comparison_df, aes(x = Actual, y = Predicted)) +
        geom_point(alpha = 0.5, color = "blue") +
        geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
        theme_minimal() +
        labs(title = "Predicted vs. Actual Prices", x = "Actual Price", y = "Predicted Price")

# Residual plot to visualise prediction errors
residuals <- test_actual_price - test_predicted_price

ggplot(data.frame(Actual = test_actual_price, Residuals = residuals), aes(x = Actual, y = Residuals)) +
        geom_point(alpha = 0.5, color = "purple") +
        geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
        theme_minimal() +
        labs(title = "Residuals vs. Actual Prices", x = "Actual Price", y = "Residuals (Actual - Predicted)")

# Combine into a data frame
comparison_df <- data.frame(
        Actual_Price = test_actual_price,
        Predicted_Price = test_predicted_price
)

# Sample 10 random rows for the table
set.seed(123)
sample_comparison <- comparison_df[sample(1:nrow(comparison_df), 10), ]

# Print table
print(sample_comparison)

ggplot(sample_comparison, aes(x = Actual_Price, y = Predicted_Price)) +
        geom_point(alpha = 0.7, color = "blue", size = 3) +
        geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
        theme_minimal() +
        labs(
                title = "Sample Comparison: Predicted vs. Actual Prices",
                x = "Actual Price",
                y = "Predicted Price"
        )
