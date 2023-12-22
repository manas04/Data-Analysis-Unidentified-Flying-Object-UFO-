# Library Installation --------------------------------------------------------------------------------------------------------------
install.packages("gridExtra")
install.packages("CatEncoders")
install.packages("corrplot")
install.packages("ggplot2")
install.packages("tm")
install.packages("wordcloud")
install.packages('dplyr')
install.packages('corrplot')
install.packages("glmnet")
install.packages("rpart")
install.packages("caret",dependencies = TRUE)
install.packages("ROCR")
install.packages("Rcpp")
install.packages("e1071",dependencies = TRUE)
install.packages("gplots")

# Library Loading -------------------------------------------------------------------------------------------------------------------
library(dplyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(glmnet) 
library(corrplot)
library(ROCR)
library(wordcloud)
library(RColorBrewer)

# Data Loading into a vector----------------------------------------------------------------------------------------------------------
ufo_data <- read.csv("~/Desktop/Fall 23/FDS-560/FDS_FINAL_PROJECT/ufo-sightings-transformed.csv")

# Some Initial Data Exploration ------------------------------------------------------------------------------------------------------
 
head(ufo_data) # Display the first 5 rows of the data
print(dim(ufo_data)) # Print the dimensions (rows, columns) of the data
print(names(ufo_data)) # Print the column names of the data
print(str(ufo_data)) # Print the structure and data types of columns
print(summary(ufo_data)) # Print summary statistics for each columns

# Data Cleaning ----------------------------------------------------------------------------------------------------------------------

colSums(is.na(ufo_data)) # Count the number of NA values in each column
data_transformed <- na.omit(ufo_data) # Remove rows with NA values
data_transformed <- distinct(data_transformed) # Remove duplicate rows

# Data Visualization------------------------------------------------------------------------------------------------

#Stacked Bar Plot for UFO Shapes Over Time
ggplot(ufo_data, aes(x = Year, fill = UFO_shape)) +
  geom_bar(position = "stack") +
  labs(title = "Stacked Bar Plot of UFO Shapes Over Years", x = "Year", y = "Count")

#Density Plot for Year
ggplot(ufo_data, aes(x = Year)) +
  geom_density(fill = "blue", alpha = 0.5) +
  labs(title = "Density Plot of UFO Sightings Over Years", x = "Year")

# Define a color palette
color_palette <- brewer.pal(8, "Dark2")

# Create the word cloud
wordcloud(words = ufo_data$UFO_shape, 
          max.words = 100, 
          random.order = FALSE, 
          rot.per = 0.35, # 35% of words are displayed at an angle
          scale = c(3, 0.5), # Scale between most and least frequent words
          colors = color_palette)

# Correlation matrix
numeric_var <- data_transformed %>% select_if(is.numeric) 
cat_var <- data_transformed %>% select_if(is.factor)
corr_matrix<-cor(numeric_var) #correlation matrix for the numerical data
corrplot(corr_matrix, method = "color", addCoef.col = "black", number.cex = 0.7)

# Create a summary count of incidents by hour
hourly_counts <- ufo_data %>%
  group_by(Hour) %>%
  summarise(Count = n()) %>%
  arrange(Hour)

# Create a summary count of incidents by month
monthly_counts <- ufo_data %>%
  group_by(Month) %>%
  summarise(Count = n()) %>%
  arrange(Month)

# Convert month number to month name for better readability
monthly_counts$Month <- month.abb[monthly_counts$Month]

# Plotting the hourly count bar plot
ggplot(hourly_counts, aes(x = Hour, y = Count)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Incident Count of Hour to Hour starting from 1906 to 2014",
       x = "Hour of the Day", y = "Incident Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Plotting the monthly count bar plot
ggplot(monthly_counts, aes(x = Month, y = Count, fill = Month)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Paired") + # Using a color palette for differentiation
  labs(title = "Incident Count of Each Month Starting From 1906 to 2014",
       x = "Month", y = "Incident Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Count shapes
shape_counts <- table(ufo_data$UFO_shape)

# Display the counts
print(shape_counts)

# Ensure 'UFO_shape' column is a factor for proper ordering in the bar plot
ufo_data$UFO_shape <- as.factor(ufo_data$UFO_shape)

# Create a bar plot for UFO shapes
ggplot(ufo_data) +
  aes(x = UFO_shape) +
  geom_bar(fill = "red", color = "black") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Number of Sightings by UFO Shape", x = "UFO Shape", y = "Count")


# Preprocess the Data for Model Building --------------------------------------------------------------------------------------------
ufo_data <- ufo_data %>%
  mutate(UFO_shape_binary = ifelse(UFO_shape %in% c("Circle", "Sphere", "Round"), 1, 0)) %>%
  select(Year, Month, Hour, Country, latitude, longitude, length_of_encounter_seconds, UFO_shape_binary)

# Bar Plot to visualize the distribution of Circular vs Non-Circular shapes
ggplot(ufo_data, aes(x = as.factor(UFO_shape_binary), fill = as.factor(UFO_shape_binary))) +
  geom_bar(show.legend = FALSE) + # Hide legend
  scale_x_discrete(labels = c("0" = "Non-Circular", "1" = "Circular")) +
  scale_fill_manual(values = c("0" = "red", "1" = "blue")) + # Assign colors to each category
  labs(title = "Distribution of Circular vs Non-Circular UFO Shapes",
       x = "UFO Shape Category",
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0))

# Dummy Variable Creation and Standardization ---------------------------------------------------------------------------------------
# Convert 'Country' to factor and create dummy variables
library(caret)
ufo_data$Country <- as.factor(ufo_data$Country)
dummy_model <- dummyVars(" ~ .", data = ufo_data)
ufo_data_transformed <- predict(dummy_model, ufo_data)

# Convert to data frame and ensure UFO_shape_binary is a column
ufo_data_transformed <- data.frame(ufo_data_transformed)
ufo_data_transformed$UFO_shape_binary <- ufo_data$UFO_shape_binary

standardization <- function(x) {
  (x - mean(x)) / sd(x)
}
ufo_data_transformed <- data.frame(lapply(ufo_data_transformed,standardization))
ufo_data_transformed$UFO_shape_binary <- ufo_data$UFO_shape_binary



# Split the data into training and testing sets--------------------------------------------------------------------------------------
set.seed(123) 
splitIndex <- createDataPartition(ufo_data_transformed$UFO_shape_binary, p = 0.8, list = FALSE)
train_data <- ufo_data_transformed[splitIndex,]
test_data <- ufo_data_transformed[-splitIndex,]


# Model Training --------------------------------------------------------------------------------------------------------------------
train_data <- train_data[, colnames(test_data)]

# Train the logistic regression model with regularization
x <- model.matrix(UFO_shape_binary ~ . - 1, data = train_data)
y <- train_data$UFO_shape_binary

# Fit the model using cross-validation
cv_model <- cv.glmnet(x, y, family = "binomial", alpha = 1)
best_lambda <- cv_model$lambda.min

# Train the final model
final_model <- glmnet(x, y, family = "binomial", lambda = best_lambda, alpha = 1)

# Make predictions on the test data
test_x <- model.matrix(UFO_shape_binary ~ . - 1, data = test_data)
predictions <- predict(final_model, newx = test_x, type = "response", s = best_lambda)
predicted_class <- ifelse(predictions > 0.5, 1, 0)

# Model Evaluation and Visualization ------------------------------------------------------------------------------------------------------------------

#ROC curve
roc_pred <- prediction(predictions, test_data$UFO_shape_binary)
roc_perf <- performance(roc_pred, "tpr", "fpr")
plot(roc_perf, col = "blue", main = "ROC Curve")

#Precision Recall Curve
pr_pred <- prediction(predictions, test_data$UFO_shape_binary)
pr_perf <- performance(pr_pred, "prec", "rec")
plot(pr_perf, col = "red", main = "Precision-Recall Curve")

# F1 Score, Precision, and Recall
predicted_class_factor <- factor(predicted_class, levels = c(0, 1))
actual_class_factor <- factor(test_data$UFO_shape_binary, levels = c(0, 1))
conf_matrix <- confusionMatrix(predicted_class_factor, actual_class_factor)
f1_score <- 2 * (conf_matrix$byClass["Precision"] * conf_matrix$byClass["Recall"]) / (conf_matrix$byClass["Precision"] + conf_matrix$byClass["Recall"])
precision <- conf_matrix$byClass["Precision"]

cat("F1 Score:", f1_score, "\n")
cat("Precision:", precision, "\n")

# Calculate and Print Model Accuracy
accuracy <- sum(predicted_class == test_data$UFO_shape_binary) / length(test_data$UFO_shape_binary)
print(paste("Accuracy:", round(accuracy, 4)))

