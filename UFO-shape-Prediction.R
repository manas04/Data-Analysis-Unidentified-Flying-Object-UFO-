

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

# All libraries for the analysis 
library(dplyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(glmnet) 
library(corrplot)
library(ROCR)


# Data Loading
ufo_data <- read.csv("C:/Users/ASUS/Downloads/ufo-sightings-transformed.csv")
head(ufo_data) #printing first 5 rows of the data 
print(dim(ufo_data)) # printing shape of the data
print(names(ufo_data)) #Printing names of the columns in the data
print(str(ufo_data)) #basic statistics of the ufo_data
print(summary(ufo_data)) #Descriptive Statistics

# Data Visualization
numeric_var <- data_transformed %>% select_if(is.numeric) 
cat_var <- data_transformed %>% select_if(is.factor)
corr_matrix<-cor(numeric_var) #correlation matrix for the numerical data
corrplot(corr_matrix, method = "color", addCoef.col = "black", number.cex = 0.7)


table(ufo_data$UFO_shape_binary)

# Preprocess the data
ufo_data <- ufo_data %>%
  mutate(UFO_shape_binary = ifelse(UFO_shape %in% c("Circle", "Sphere", "Round"), 1, 0)) %>%
  select(Year, Month, Hour, Country, latitude, longitude, length_of_encounter_seconds, UFO_shape_binary)

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
head(ufo_data_transformed)

# Split the data into training and testing sets
set.seed(123) # for reproducibility
splitIndex <- createDataPartition(ufo_data_transformed$UFO_shape_binary, p = 0.8, list = FALSE)
train_data <- ufo_data_transformed[splitIndex,]
test_data <- ufo_data_transformed[-splitIndex,]

# Ensure both datasets have the same columns
train_data <- train_data[, colnames(test_data)]

# Train the logistic regression model with regularization (Lasso)
x <- model.matrix(UFO_shape_binary ~ . - 1, data = train_data) # -1 to exclude the intercept
y <- train_data$UFO_shape_binary

# Fit the model using cross-validation
cv_model <- cv.glmnet(x, y, family = "binomial", alpha = 1) # alpha = 1 for Lasso
best_lambda <- cv_model$lambda.min

# Train the final model
final_model <- glmnet(x, y, family = "binomial", lambda = best_lambda, alpha = 1)

# Make predictions on the test data
test_x <- model.matrix(UFO_shape_binary ~ . - 1, data = test_data)
predictions <- predict(final_model, newx = test_x, type = "response", s = best_lambda)
predicted_class <- ifelse(predictions > 0.5, 1, 0)

# Evaluate the model
cm <- confusionMatrix(factor(predicted_class), factor(test_data$UFO_shape_binary))
# Summary of the model
print(summary(cv_model))

# Visualization of answer
 #ROC curve
roc_pred <- prediction(predictions, test_data$UFO_shape_binary)
roc_perf <- performance(roc_pred, "tpr", "fpr")
plot(roc_perf, col = "blue", main = "ROC Curve")

#Precision Recall Curve
pr_pred <- prediction(predictions, test_data$UFO_shape_binary)
pr_perf <- performance(pr_pred, "prec", "rec")
plot(pr_perf, col = "red", main = "Precision-Recall Curve")


