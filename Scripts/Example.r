# Include Libs

# Import CSV file
AirbnbData <- read.csv("DataSet/AB_NYC_2019.csv")

# Create Train and Test subsets
set.seed(25)
train_indices <- sample(seq_len(nrow(AirbnbData)), size = 0.7 * nrow(AirbnbData))

training <- AirbnbData[train_indices, ]
testing <- AirbnbData[-train_indices, ]

