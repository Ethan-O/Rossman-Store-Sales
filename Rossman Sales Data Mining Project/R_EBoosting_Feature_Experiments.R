#install.packages('xgboost')
#install.packages('readr')
#install.packages('dplyr')
#install.packages('TTR')
#install.packages('caret')
#install.packages('ggplot2')
library(readr)
library(xgboost)
library(dplyr)
library(TTR)
library(lubridate)
library(caret)
library(ggplot2)

cat("reading the train and test data\n")
train <- read_csv("input/train.csv")
test  <- read_csv("input/test.csv")
store <- read_csv("input/store.csv")



# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- merge(train,store)
test <- merge(test,store)


# There are some NAs in the integer columns so conversion to zero
# the test na for open are conveted to 1
train[is.na(train)]   <- 0
test[is.na(test$Open),"Open"] <- 1
test[is.na(test)]   <- 0

#Converts a vecor of with na's to ones
 one_na <-function(x) {
   x[is.na(x) == TRUE] <- 1
   return(x)
 }
 
#Creates a vector that counts the number of trailing zeros at the index of the inputed vector
trailing_zeros<- function(x){
  current_zeros =0
  num_zeros = c()
  for( i in 1:length(x)){
    num_zeros = c(num_zeros,current_zeros)
    if( x[i] ==0){
      current_zeros = current_zeros + 1
    }
    else{
      current_zeros = 0
    }
  }
  return(num_zeros)
}

#Calculates the leading zeros at each index of the inputed function and returns it as a vector
leading_zeros<-function(x){
  current_zeros =0
  num_zeros = c()
  for( i in length(x):1){
    num_zeros = c(current_zeros,num_zeros)
    if( x[i] ==0){
      current_zeros = current_zeros + 1
    }
    else{
      current_zeros = 0
    }
  }
  return(num_zeros)
}
 

 
#Engineers the the LongClose and LongOpen features 
#LongClose: assigns the number of days the store will be closed the day before the first close
#LongOpen: assigns the number of days the store was closed on the reopen day

train.C <- train %>%
  group_by(Store) %>%
  arrange(as.Date(Date,"%Y-%m-%d")) %>%
  mutate(Opened = zero_na((Open - lag(Open)) >0),
         Closed = zero_na((Open - lag(Open)) <0 )) %>%
  mutate(TomorrowClosed = zero_na(lead(Closed))) %>%
  mutate(LongClose= TomorrowClosed*leading_zeros(one_na(Open)),
         LongOpen= Opened*trailing_zeros(one_na(Open)))


  test.C <- test %>%
    group_by(Store) %>%
    arrange(as.Date(Date,"%Y-%m-%d")) %>%
    mutate(Opened = zero_na((Open - lag(Open)) >0),
           Closed = zero_na((Open - lag(Open)) <0 )) %>%
    mutate(TomorrowClosed = zero_na(lead(Closed))) %>%
    mutate(LongClose= TomorrowClosed*leading_zeros(one_na(Open)),
           LongOpen= Opened*trailing_zeros(one_na(Open)))



train <- train.C
test <- test.C

#Only training on the stores that are in the test set
#Also only on rows that are open and have sales
train <-train[train$Store %in% unique(test$Store),]
train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]
names(train)



train.MA <- train %>%
  group_by(Store) %>%
  arrange(as.Date(Date,"%Y-%m-%d")) %>%
  mutate(S_MAQ =runMean(Sales,364/(4)),
         C_MAQ = runMean(Customers,364/(4)),
         S_MA300 = runMean(Sales,364*1),
         C_MA300 = runMean(Customers,364*1))

train <- train.MA
# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))
train$day <- as.integer(format(train$Date, "%d"))


# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
#train <- train[,-c(3,8)]
train <- train[,-c(3,8)]

# seperating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$day <- as.integer(format(test$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
#test <- test[,-c(4,7)]
test <- test[,-c(4,7)]

#ind_features = c("Promo","SchoolHoliday","Promo2","Promo2SinceWeek","Promo2SinceYear","PromoInterval")
#train <- train[,!names(train) %in% ind_features]

c("S_MAQ","C_MAQ" ,"S_MA300" ,"C_MA300") 

#CHOOSE FEATURE SET HERE
#The features that are removed for the: complete, MA, reopen, and baseline models
complete_feat = c("TomorrowClosed","Opened","Closed","Sales","Customers")
ma_feat = c("TomorrowClosed","LongClose","LongOpen","Opened","Closed","Sales","Customers")
reopen_feat = c("S_MAQ","C_MAQ" ,"S_MA300" ,"C_MA300","TomorrowClosed","Opened","Closed","Sales","Customers")
base_feat = c("S_MAQ","C_MAQ" ,"S_MA300" ,"C_MA300","TomorrowClosed","LongClose","LongOpen","Opened","Closed","Sales","Customers")

feature.names = names(train)[!(names(train) %in% base_feat)]

cat("Feature Names\n")
feature.names

#Replacing categorical variables with numeric labels
cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

#training data
tra<-train[,feature.names]
#Root Mean Percent Square Error metric
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}
#Randomly sampling development data indicies
h<-sample(300000:nrow(train),10000)

#Log scaling the sales
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)

#Params where picked from the orginal template
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.02, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.3, # 0.7
                colsample_bytree    = 0.7 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 3200, #300, #280, #125, #250, # changed from 300
                    verbose             = 1,
                    print_every_n       = 100,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)



#Validation set predictions and accuracy metrics
pred_v <- exp(predict(clf, dval)) -1
postResample(train$Sales[h],pred_v)


#DON'T RUN THIS BLOCK IF NOT USING MOVING AVERAGES
#This adds the last moving avererages to each test row
t_ma <- train %>%
  group_by(Store) %>%
  filter(row_number()==n()) %>%
  select(S_MAQ,C_MAQ,S_MA300,C_MA300)

test <-merge(test,t_ma,by= "Store")

#Block Ends

#Making predictions from test data and writing to csv
pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
submission <- data.frame(Id=test$Id, Sales=pred1)
cat("saving the submission file\n")
write_csv(submission, "None.csv")


#Plots
#Devlopment Data goodness of fit plot
eval = data.frame(Sales = train[h,"Sales"],Pred.Sales = pred_v)
ggplot(eval,aes(y=Sales,x=Pred.Sales))+ geom_point(color="darkblue")  + geom_abline(color = "red")


#Importences
library(ggthemes)
importance <- xgb.importance(feature_names = feature.names, model = clf)

ggplot(importance,aes(x = reorder(importance$Feature,-Gain), y = Gain))+ geom_bar(stat = "identity",fill="darkblue") +
  
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab("Features") + ggtitle("XGBoost Feature Importances")
