setwd("C:/Users/jjensen/Dropbox/R/Coursera - Data Science Certificate/8 - Practical Machine Learning/project")


######################
##
## Practical Machine Learning
## Predicion Assignment
## Author: Joshua Jensen
##
## Objective: Predict  ('classe' variable) from FitBit sensor data
##
######################


library(dplyr)
library(ggplot2)
library(readr)
library(caret)
#library(rattle)

set.seed(221)

df <- read_csv("pml-training.csv")
eval_test <- read_csv("pml-testing.csv")



## Filter down to only applicable measures for the assignment
## ie. vairables that are present in both pml-training.csv & pml-testing.csv

# seed with the "[EMPTY]" column name
invalid_columns <- "[EMPTY]"
for (i in 1:length(colnames(eval_test))) {
  if(sum(is.na(eval_test[,i]))==nrow(eval_test)){
    temp <- colnames(eval_test)[i]
    
    invalid_columns <- c(invalid_columns,temp)
  }
}

df <- select(df, -one_of(invalid_columns))
eval_test <- select(eval_test, -one_of(invalid_columns))


## Define the models training and testing sets
## We will use a K-Folds approach with 5 folds to cross validate, reduce bias, but maintain stable variance

# folds <- createFolds(df$classe, k = 5)
# 
# sapply(folds, length)
# 
# 
# training <- df[-folds$Fold1,]
# testing <- df[folds$Fold1,]
# 
# lapply(training, class)


in_train <- createDataPartition(y=df$classe,
                               p=0.7, 
                               list=FALSE)
training <- df[in_train,]
testing <- df[-in_train,]

## Exploratory analysis, plotting predictors

colnames(training)

lapply(training, class)

types <- NULL
for(i in 1:ncol(training)){
  temp <- lapply(training, class)[[i]]
  types <- c(types,temp)
}
col_types <- data.frame(colnames(training),types)

table(col_types$types)


# density plots & box plots for all variables
for(i in 1:ncol(training)){
  if( lapply(training, class)[[i]] %in% c("integer","numeric")){
    # create density ggplot
    temp_plot <- ggplot(training, aes_string(x= colnames(training)[i], colour = "classe")) + 
      geom_density() +
      ggtitle(paste(sprintf("%02d",i), "-", colnames(training)[i], "- Density"))
    
    print(temp_plot)
    ggsave(temp_plot, filename = paste0("./plots/", sprintf("%02d",i), "_", colnames(training)[i], "_density.png"))
    
    # create boxplot ggplot
    temp_plot <- ggplot(training, aes_string(x = "classe", y = colnames(training)[i])) + 
      geom_boxplot(aes(fill = classe)) + 
      geom_jitter(aes(alpha = .01)) +
      ggtitle(paste(sprintf("%02d",i), "-", colnames(training)[i], "- Boxplot"))
    
    print(temp_plot)
    ggsave(temp_plot, filename = paste0("./plots/", sprintf("%02d",i), "_", colnames(training)[i], "_box.png"))
  }
}

#
### Data Transformations (run on test set too)
#

## filter out the bs stuff
irrelevant_columns <- c("user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window" )
training <- select(training, -one_of(irrelevant_columns))

training$classe <- as.factor(training$classe)
class(training$classe)

table(training$classe)
table(training$classe)/nrow(training)

# basic desicion tree
model_fit_tree <- train(classe ~ .,
                   data = training,
                   method = "rpart")

fancyRpartPlot(model_fit_tree$finalModel)

confusionMatrix(training$classe, predict(model_fit_tree, newdata = training))

confusionMatrix(testing$classe, predict(model_fit_tree, newdata = testing))


## random forest
model_fit_rf <- train(classe ~ .,
                   data = training,
                   method = "rf",
                   prox = TRUE)



confusionMatrix(training$classe, predict(model_fit_rf, newdata = training))

confusionMatrix(testing$classe, predict(model_fit_rf, newdata = testing))


# ## run pca
# pca_training <- preProcess(select(training,-classe),
#                         method="pca",
#                         thresh = .95)
# p_pca_training <- predict(pca_training, newdata = select(training,-classe))
# 
# 
# ## random forest with pca
# model_fit_rf_pca <- train(training$classe ~ .,
#                       data = p_pca_training,
#                       method = "rf",
#                       prox = TRUE)
# 
# confusionMatrix(training$classe, predict(model_fit_rf_pca, newdata = p_pca_training))
# 
# confusionMatrix(testing$classe, predict(model_fit_tree, newdata = testing))


## predict against eval_test

predictions_rf <- predict(model_fit_rf, newdata = eval_test)

write.csv(data.frame( eval_test, predictions_rf),"predictions.csv")


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./predictions/","problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictions_rf)


save(model_fit_rf, file = "model_fit_rf.RData")
save(model_fit_tree, file = "model_fit_tree.RData")
