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
setwd("/Users/davide_lupis/Desktop/UniversityMaterial/R-statistical Learning/Project/")

########################################################


#################### Data Preparation ##################

data <- read.csv("MBTI 500.csv")
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
   ###############################################
   CV_score[label]  <- vector_score
   
   # Plot
   #plot(CV_score[label], main = label, xlab = 'Kfolds',ylab = 'Balanced Accuracy')
   #lines(CV_score[label])
}

#######################################################
# load("Model_Energy_1.Rda")
#inizio 20:15 
# fine 15:15
#write.csv(CV_score,"CV_score.csv", row.names = FALSE,sep = ',')


# init plot
par(mfrow=c(2,2))
for (label in labels){
   # Plot
   plot(CV_score[,label], 
        ylim = c(0.70,0.95),
        main = label, 
        frame = FALSE, pch = 19,
        lty = 1, lwd = 1,
        xlab = 'Kfolds',ylab = 'Balanced Accuracy',col='blue')
   lines(CV_score[label])
   text( 1:5, CV_score[,label]+0.05 ,round(CV_score[,label],3) )
}
# for each test row
# transorm tf_idf_vectorizer for each model 
# random forest for each tf_idf_vectorizer
# combine predictions
# issue slow tfidf , sample the type ENTJ



############## Final Model ######################
library(tm)
library(superml)
library(caret)
library(dplyr)
library(MASS)
library(randomForest)
library(ggplot2)

setwd("/Users/davide_lupis/Desktop/UniversityMaterial/R-statistical Learning/Project/")
data <- read.csv("MBTI 500.csv")
# Definition of a subproblem
data['Energy'] <- as.factor(ifelse(grepl("E",data$type),1,0))
data['Information'] <- as.factor(ifelse(grepl("N",data$type),1,0))
data['Decision'] <- as.factor(ifelse(grepl("T",data$type),1,0))
data['Organize'] <- as.factor(ifelse(grepl("J",data$type),1,0))
data$type <- as.factor(data$type)
labels <-  c('Energy','Information','Decision','Organize','type')

table(data$type)
# Create a Downsized dataset
data_downsized <- data %>% 
   group_by(type) %>% 
   sample_n(150,replace = TRUE)
# Shuffle the dataframe by rows
data_downsized <- data_downsized[sample(1:nrow(data_downsized)), ]
table(data_downsized$type)
# Create a TEST  dataset with unseen data 
test_downsized <- data[!(as.character(data$posts) %in% as.character(data_downsized$posts)), ]
# Create a Downsized dataset
test_downsized <- data %>% 
   group_by(type) %>% 
   sample_n(50,replace = TRUE)
# Shuffle the dataframe by rows
test_downsized <- test_downsized[sample(1:nrow(test_downsized)), ]
table(test_downsized$type)
table(data_downsized$type)
# TfIdfVectorizer text for training data 
vocabulary <- 400
tfv <- TfIdfVectorizer$new(max_features = vocabulary,remove_stopwords = TRUE)
# Fit TFIDF on train
tfv$fit(data_downsized$posts)


# Transform train  using fitted TFIDF on Train Data
train_tf_features <- data.frame(
   tfv$transform(train$posts)
)

# TFIDF on Test using the previously fitted TFIDF
test_tf_features <- data.frame(
   tfv$transform(test_downsized$posts)
)

# Save TRAIN and TEST TFIDF 
#write.csv(train_tf_features )

#label <- 'Organize'
# train model for each label
for (label in labels){
   # Crearte a dataframe
   train <- data_downsized
   # Set the label
   train["label"] <- train[ ,label]
   
  
   # set label features
   train_tf_features['label'] <-  train[,label] 
   
   print(table(train_tf_features['label']))

   # Fit Model on Train 
   model <- randomForest(label ~., data = train_tf_features) 
   
  
   
   
   
   test_tf_features['label'] <-  test_downsized[,label]
   print(table(test_tf_features['label']))

   
   # Predict
   prediction_label <-  paste('predicted_',label,sep = '')
   test_tf_features[ ,prediction_label ] <-  predict(model, newdata = test_tf_features)
   
   # Confusion Matrix and Scores
   cm <- confusionMatrix(test_tf_features[ ,prediction_label ],
                         test_tf_features[ ,'label'] 
                         , mode = "everything")
   
   score <- cm$byClass['Balanced Accuracy']

   #Save Model
   info <- '_RandomForest_'
   score_round <- round(score,2)*100
   save(model, 
        file= paste(label,info,score_round,'.Rda',sep = '')
   )  
   
   # Collect predictions
   test_downsized[,prediction_label] <- test_tf_features[ ,prediction_label ]
}

# check the complete profile
test_downsized['predicted_letter_Energy'] <-  as.factor(ifelse(test_downsized$predicted_Energy== 1,'E','I'))
test_downsized['predicted_letter_Information'] <-  as.factor(ifelse(test_downsized$predicted_Information== 1,'N','S'))
test_downsized['predicted_letter_Decision'] <-  as.factor(ifelse(test_downsized$predicted_Decision ==1,'T','F'))
test_downsized['predicted_letter_Organize'] <-  as.factor(ifelse( test_downsized$predicted_Organize ==1,'J','P'))
test_downsized['predicted_type'] <-  predict(model, newdata = test_tf_features)

# Combine subpredictions
test_downsized['predicted_type_combined'] <- paste(test_downsized$predicted_letter_Energy,
                                          test_downsized$predicted_letter_Information,
                                          test_downsized$predicted_letter_Decision,
                                          test_downsized$predicted_letter_Organize
                                           ,sep = '')
# FINAL ACCURACY 100 type each
result_accuracy_type <-  sum(test_downsized['type'] == test_downsized['predicted_type'] )/dim(test_downsized)[1]
test_downsized['predicted_correctly'] <- test_downsized['type'] == test_downsized['predicted_type']
result_accuracy_type

result_accuracy <-  sum(test_downsized['type'] == test_downsized['predicted_type_combined'] )/dim(test_downsized)[1]
result_accuracy

names(test_downsized)
# Summary on data
table(data$type)
table(data_downsized$type)
table(test_downsized$type)

# How and where do we make mistakes ? 
sort(table(test_downsized['predicted_type_combined'])-table(test_downsized['type']))
sort(table(test_downsized['predicted_type'])-table(test_downsized['type']))


table(test_downsized['predicted_type_combined'])-table(test_downsized['type'])
table(test_downsized['predicted_type'])-table(test_downsized['type'])
test_downsized[,c('predicted_type','type')]

#
"Discussion. 
Given that some types are rare, using the recursive model allow us to have balanced data and get almost equivalent performances 77 vs 80 
proportion of data
ENFJ  ENFP  ENTJ  ENTP  ESFJ  ESFP  ESTJ  ESTP  INFJ  INFP  INTJ  INTP  ISFJ  ISFP  ISTJ  ISTP 
 1534  6167  2955 11725   181   360   482  1986 14963 12134 22427 24961   650   875  1243  3424 
 
 Update :
 If i use the type as label i can just train on 150 , if i use the recursive model i have 2400 data for each label
"