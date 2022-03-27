#install.packages("tm")

library(tm)
library(superml)
library(caret)
library(dplyr)
library(MASS)
library(randomForest)
library(ggplot2)

#https://www.truity.com/myers-briggs/about-myers-briggs-personality-typing
"The Myers-Briggs Type Indicator (MBTI) is a psychological assessment tool used to classify people into one of 16 different personality types. 
The classification system consists of a four-letter code based on four dimensions, where each letter in the code refers to the predominant trait in each dimension. 
The four dimensions are:

    Extraversion vs. Introversion: How do you gain energy? Extraverts like to be with others and gain energy from people and the environment. Introverts gain energy from alone-time and need periods of quiet reflection throughout the day.
    Sensing vs. Intuition: How do you collect information? Sensors gather facts from their immediate environment and rely on the things they can see, feel and hear. Intuitives look more at the overall context and think about patterns, meaning, and connections.
    Thinking vs. Feeling: How do you make decisions? Thinkers look for the logically correct solution, whereas Feelers make decisions based on their emotions, values, and the needs of others.
    Judging vs. Perceiving: How do you organize your environment? Judgers prefer structure and like things to be clearly regulated, whereas Perceivers like things to be open and flexible and are reluctant to commit themselves.
"

##################### Data #############################
setwd("/Users/davide_lupis/Desktop")
data <- read.csv("MBTI 500.csv")
########################################################


#################### Data Preparation ##################


# Definition of a subproblem
data['Energy'] <- as.factor(ifelse(grepl("E",data$type),1,0))
data['Information'] <- as.factor(ifelse(grepl("N",data$type),1,0))
data['Decision'] <- as.factor(ifelse(grepl("T",data$type),1,0))
data['Organize'] <- as.factor(ifelse(grepl("J",data$type),1,0))
########################################################



################ Model Pipeline ##################

# For each subproblem:
  # Create a Cross validation
    # Create a label , assumption of mutually exclusive labels even if they are more soft classification
    # Separate data as test and train, Balance-Shuffle Train and  Downsize-Shuffle Test
    # Encode Text with TF-IDF 
    # Fit Model
    # Test Model 
    # Collect F1Score



# Create a Vector for the performance of each submodel
k_folds <- 5
CV_score <- data.frame( matrix(0, 5, 4) )
labels <-  c('Energy','Information','Decision','Organize')
colnames(CV_score) <- labels
#label <- labels[1]
# init plot
par(mfrow=c(2,2))
for (label in labels){
   # Crearte a dataframe
   df <- data
   # Set the label
   df["label"] <- df[ ,label]
   ############### Create a CV ###################
   
   # For each folder
   vector_score <- rep(0,k_folds)
   for (k in 1:k_folds){
        
         # Set number of sample in train 
         record_train_per_class <- 2000
         
         #+++++++Create a TRAIN dataset++++++++++++++++++++++++++
         # Group by label and sample the same quantity, resampling if not possible ( unlikely )
         train <- df %>% 
            group_by(label) %>% 
           sample_n(record_train_per_class,replace = TRUE)
         # Define Features and Labels
         train <- train[,c("posts","label")]
         # Shuffle the dataframe by rows
         train <- train[sample(1:nrow(train)), ]
         #++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         
         
         #+++++++Create a TEST  dataset with unseen data ++++++ 
         test <- df[!(as.character(df$posts) %in% as.character(train$posts)), ]
         
         # Define Features and Labels
         test <- test[,c("posts","label")]
         
         # Downsize Balanced test 
         test <- test %>% 
           group_by(label) %>% 
           sample_n(record_train_per_class,replace = TRUE)
         # Shuffle the dataframe by rows
         test <- test[sample(1:nrow(test)), ]
         #++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         
         
         # TfIdfVectorizer
         vocabulary <- 500
         tfv <- TfIdfVectorizer$new(max_features = vocabulary,
                                    remove_stopwords = TRUE)
        
         # Fit TFIDF on train
         tfv$fit(train$posts)
         
         # Transform train  using fitted TFIDF on Train Data
         train_tf_features <- data.frame(
           tfv$transform(train$posts)
         )
         # set label features
         train_tf_features['label'] <-  train$label
         
         # Fit Model on Train 
         model <- randomForest(label ~., data = train_tf_features,) 
         
         # Eval Model 
         actual <- test$label
         
         # TFIDF on Test using the previously fitted TFIDF
         test_tf_features <- data.frame(
             tfv$transform(test$posts)
         )
         
         train_tf_features['label'] <-  test$label
         
         # Predict
         predicted<- predict(model, newdata = test_tf_features)
         
         # Confusion Matrix and Scores
         cm <- confusionMatrix(predicted, actual, mode = "everything", positive="1")
         score <- cm$byClass['Balanced Accuracy']
         # Register the score for each folder
         vector_score[k] <- score
         
         #Save Model
         info <- 'RandomForestK'
         score_round <- round(score,2)*100
         save(model, 
              file= paste(label,k,info,score_round,'.Rda',sep = '')
              )  
   }
   
   CV_score[label]  <- vector_score
   
   # Plot
   plot(CV_score[label], main = label, xlab = 'Kfolds',ylab = 'Balanced Accuracy')
   lines(CV_score[label])
}

#######################################################
# load("Model_Energy_1.Rda")




