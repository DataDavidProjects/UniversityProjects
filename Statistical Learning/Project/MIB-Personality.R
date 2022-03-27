
####################### PACKAGES###########################
#install.packages("superml", dependencies=TRUE)
#install.packages('word2vec')
#install.packages("wordcloud2")
#install.packages("RColorBrewer")

#install.packages("tm")
library(tm)
library(wordcloud)
library(RColorBrewer)

library(superml)
library(tidyverse)
library(dplyr)
library(MASS)
library(randomForest)
library(ggplot2)
###########################################################



################ PIPELINE PSEUDO CODE #####################
# Data are taken from  twitter and got cleaned with text and stopwords
# Twitter - last N post - Model - Psychograph segmentation and REngine.


# Scores: []
# For 5 times 
#   step 1 : create a balanced sample train 1600, 100 for each personality
#          and test 160, 10 record for each personality in test
#   step 2 : fit the model  and evaluate it on test, metric f1/accurary
#   step 3 : save and append score in Scores
###########################################################



##################### FULL PIPELINE  ###################
setwd("C:/Users/david/Desktop")
data.all <- read.csv("MBTI 500.csv",encoding = 'Latin-1')
#data.all$posts<-str_replace_all(data.all$posts,"[^[:graph:]]", " ") 


names(data.all)[2] <- "label"
data.all$label <- as.factor(data.all$label)

# WordCloud for each MIB type
# for (type in  unique(data.all$label)  ){
#   
#   MIB_type <- data.all[ sample(which(data.all$label == type),100) , "posts" ]
#   
#   corpus <- Corpus(VectorSource(MIB_type ))
#   
#   tdm <- TermDocumentMatrix(corpus,
#                             control = list(
#                             wordLengths=c(0,Inf),
#                             removePunctuation = TRUE,
#                             stopwords = stopwords("english"),
#                             removeNumbers = TRUE, tolower = TRUE) 
#                             
#                             )
#   m <- as.matrix(tdm)
#   v <- sort(rowSums(m) ,decreasing = TRUE )
#   d <- data.frame(word = names(v) , freq= v)
#  
#   jpeg( paste(  type    ,"plot.jpg",sep = "_")   )
#   wordcloud( d$word, d$freq ,
#              min.freq = 1,
#              max.words=100, 
#              random.order=FALSE, 
#              rot.per=0.35,
#              colors=rep_len( c("purple","deepskyblue4"), nrow(d) ) 
#              )
#   dev.off()
# 
# }#



########################################## MODEL ############################################

# Helper Variables
total.row <- dim(data.all)[1]
total.min_class_row <- min(table(data.all$label))
total.record_train_per_class <- 500
total.k_folds <- 10
# Cross validation over k folds
CV_score <- rep(0,total.k_folds)
for (i in 1:total.k_folds){
  
  #____________Create balanced train__________________
  train <- data.all %>% 
    group_by(label) %>% 
    sample_n(total.record_train_per_class,replace = TRUE)
  
  
  # shuffle the dataframe by rows
  train <- train[sample(1:nrow(train)), ]
  #___________________________________________________
  
  
  
  #______________ Create Test Balanced _______________
  
  # Create test, used unseen index data
  test <- data.all[!(as.character(data.all$posts) %in% as.character(train$posts)), ]
  
  # Downsize balanced test 
  test <- test %>% 
    group_by(label) %>% 
    sample_n(50,replace = TRUE)
  # shuffle the dataframe by rows
  test <- test[sample(1:nrow(test)), ]
  #___________________________________________________
  
  
  #__________ Model Training  and Feature ENG ________
  
  
  # Dimension of vocabulary,they will be the features of the model
  vocabulary <- 1000
  
  # min_df as indirect vocabulary size , es:0.5
  
  # Init TfIdfVectorizer
  tfv <- TfIdfVectorizer$new(max_features = vocabulary,
                             remove_stopwords = TRUE)
  

  # Fit on train
  tfv$fit(train$posts)
  
  # Transform train 
  train_tf_features <- data.frame(
    
      tfv$transform(train$posts)
    
  )
  
  # Apply PCA doe Dimensionality reduction and
  pca <- prcomp(train_tf_features, scale. = T)
  
  # Explained variance from pca
  var_explained <- pca$sdev^2 / sum(pca$sdev^2)
 
  # Dimensionality reduction to explain target variance 
  target_variance_explained <- 0.98
  reduced_dim <-  which(cumsum(var_explained) > target_variance_explained)[1]
  #reduced_dim <- 300 totali 
   
   # Get scores of PCA, they will be the new features
  train_tf_features <- pca$x
   
  # Reduced dimension
  train_tf_features <- train_tf_features[ ,1:reduced_dim]
  
  # Final Features and Label train set
  train <- cbind.data.frame(train_tf_features,train$label)
  colnames(train)[ length(colnames(train) )] <- "label"
  

  # Fit model 
  model <- randomForest(label ~., data = train ) 
  
  #___________________________________________________
  
  
  #_____________ Model Evaluation  __________________
  actual <- test$label
  
  # Features Engineering SCALE(test_tf_features)
  test_tf_features <- data.frame(
    scale( 
      tfv$transform(test$posts)
    )
    
  )
  
  
  # Get scores of test using pca fitted on train
  test_tf_features <- predict( pca,test_tf_features )[ ,1:reduced_dim]
  
  # Predict
  predicted<- predict(model, newdata = test_tf_features)
  
  
  #____________ Confusion Metrics  _____________
  cm = as.matrix(table(Actual = actual, Predicted = predicted))
  # Metrics
  rowsums <- apply(cm, 1, sum)
  colsums <- apply(cm, 2, sum)
  diag <- diag(cm)
  
  precision <- diag / colsums 
  recall <- diag / rowsums 
  f1 <- 2 *  ( ( precision * recall ) / ( precision + recall ) )
  data.frame(precision, recall, f1)
  
  macroPrecision <- mean(precision)
  macroRecall <- mean(recall)
  macroF1 <- mean(f1)
  data.frame(macroPrecision, macroRecall, macroF1)
 
  
  # Append Score
  CV_score[i] <- macroF1
  #___________________________________________________
  
  
  #___________________________________________________
  
}


#________________ PLOT PERFORMANCE _______________________

par(mfrow=c(1,2))
plot(CV_score , xlab = "K fold",ylab = "Macro F1 Score",
     main = paste("Model Avg Performance :",as.character( round(mean(CV_score,na.rm = T),2))) ,
     cex=1.5,type="bar",ylim = c(0.1,0.99)) 
abline(h = mean(CV_score), col = 'gray')
text( 1:10 , CV_score + 0.05 , round(CV_score,2 ) , cex = 0.7)
abline(h = max(CV_score), col = 'gray')
abline(h = mean(CV_score), col = 'gray')
abline(h = min(CV_score), col = 'gray')

#________________________________________________________

##########################################################



# TODO 
# migliorare performance 
# perche INFJ e nan


###################################################################################################
