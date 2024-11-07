# Load necessary libraries

library(ggplot2)
library(dplyr)

# Import CSV file
data <- read.csv("DataSet/AB_NYC_2019.csv")

# Filter data for Manhattan
manhattan_data <- filter(data, neighbourhood_group == "Manhattan")

# Calculate the limits for equal scaling
x_range <- range(manhattan_data$longitude)
y_range <- range(manhattan_data$latitude)
max_range <- max(diff(x_range), diff(y_range))
center_x <- mean(x_range)
center_y <- mean(y_range)

# Define equal limits
equal_xlim <- c(center_x - max_range / 2, center_x + max_range / 2)
equal_ylim <- c(center_y - max_range / 2, center_y + max_range / 2)

# Create the plot for Manhattan with fixed, equal axis limits
plot <- ggplot(manhattan_data, aes(x = longitude, y = latitude)) +
  geom_point(aes(color = cut(price, breaks = c(-Inf, 100, 200, 500, Inf))),
             alpha = 0.7, size = 0.25) +
  scale_color_manual(values = c("green", "orange", "red", "black")) +
  labs(title = "Manhattan Location Visualization by Price",
       x = "Longitude",
       y = "Latitude") +
  theme_minimal() +
  coord_fixed(ratio = 1, xlim = equal_xlim, ylim = equal_ylim) +
  theme(legend.position = "none")

# Save the plot with higher resolution
ggsave("Scripts/Price In Manhattan/plot.png", plot, width = 10, height = 8, dpi = 600, bg = "white")
