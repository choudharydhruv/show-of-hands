library( randomForest )
library( e1071 )
library( caTools )
library( ROCR )
library( mice )
library(caret)


#Load training data and convert all missing values to NA 
#Read the csv with the na.strings and strings as factors (which will pull in all the character 
#responses as text only rather than factors...this is required for subsequent operations
traindata = read.csv( "train.csv", na.strings=c("","NA"), stringsAsFactors=FALSE)

# which columns are not factors?
cols = colnames( traindata )
for ( i in 1:length( cols )) {
  col_class = class( traindata[,i] )
  if ( col_class != 'factor' ) {
    cat( cols[i], col_class, "\n" )
  }
}


#Filling in missing values using mice
set.seed(144)
#traindata_imp <- traindata
#vars.for.imputation = setdiff(names(traindata), "Happy")
#imputed = complete(mice(traindata[vars.for.imputation]))
#traindata_imp[vars.for.imputation] = imputed

#traindata <- traindata_imp

# clean-up

traindata$Happy = as.factor( traindata$Happy )

#Scaling down values
traindata[,9:109][traindata[,9:109] == "Yes"] = 1
traindata[,9:109][traindata[,9:109] == "No"] = -1
traindata[,9:109][is.na(traindata[,9:109])] = 0

traindata$Q98059 = ifelse(traindata$Q98059=="Yes",1,ifelse(traindata$Q98059=="Only-child",-1,traindata$Q98059))
#Or
#traindata$Science_or_Art[traindata$Science_or_Art=="Art"]=1;
#traindata$Science_or_Art[traindata$Science_or_Art=="Science"]=-1;
traindata$Q99982 = ifelse(traindata$Q99982=="Check!",1,ifelse(traindata$Q99982=="Nope",-1,traindata$Q99982))
traindata$Q101162 = ifelse(traindata$Q101162=="Optimist",1,ifelse(traindata$Q101162=="Pessimist",-1,traindata$Q101162))
traindata$Q101163 = ifelse(traindata$Q101163=="Dad",1,ifelse(traindata$Q101163=="Mom",-1,traindata$Q101163))
traindata$Q102089 = ifelse(traindata$Q102089=="Rent",1,ifelse(traindata$Q102089=="Own",-1,traindata$Q102089))
traindata$Q106997 = ifelse(traindata$Q106997=="Yay people!",1,ifelse(traindata$Q106997=="Grrr people",-1,traindata$Q106997))
traindata$Q108342 = ifelse(traindata$Q108342=="Online",1,ifelse(traindata$Q108342=="In-person",-1,traindata$Q108342))
traindata$Q108855 = ifelse(traindata$Q108855=="Yes!",1,ifelse(traindata$Q108855=="Umm...",-1,traindata$Q108855))
traindata$Q108856 = ifelse(traindata$Q108856=="Socialize",1,ifelse(traindata$Q108856=="Space",-1,traindata$Q108856))
traindata$Q108950 = ifelse(traindata$Q108950=="Cautious",1,ifelse(traindata$Q108950=="Risk-friendly",-1,traindata$Q108950))
traindata$Q110740 = ifelse(traindata$Q110740=="Mac",1,ifelse(traindata$Q110740=="PC",-1,traindata$Q110740))
traindata$Q111580 = ifelse(traindata$Q111580=="Supportive",1,ifelse(traindata$Q111580=="Demanding",-1,traindata$Q111580))
traindata$Q113583 = ifelse(traindata$Q113583=="Tunes",1,ifelse(traindata$Q113583=="Talk",-1,traindata$Q113583))
traindata$Q113584 = ifelse(traindata$Q113584=="People",1,ifelse(traindata$Q113584=="Technology",-1,traindata$Q113584))
traindata$Q114386 = ifelse(traindata$Q114386=="TMI",1,ifelse(traindata$Q114386=="Mysterious",-1,traindata$Q114386))
traindata$Q115777 = ifelse(traindata$Q115777=="Start",1,ifelse(traindata$Q115777=="End",-1,traindata$Q115777))
traindata$Q115899 = ifelse(traindata$Q115899=="Circumstances",1,ifelse(traindata$Q115899=="Me",-1,traindata$Q115899))
traindata$Q116197 = ifelse(traindata$Q116197=="A.M.",1,ifelse(traindata$Q116197=="P.M.",-1,traindata$Q116197))

traindata$Q116881 = ifelse(traindata$Q116881=="Happy",1,ifelse(traindata$Q116881=="Right",-1,traindata$Q116881))
traindata$Q117186 = ifelse(traindata$Q117186=="Hot headed",1,ifelse(traindata$Q117186=="Cool headed",-1,traindata$Q117186))
traindata$Q117193 = ifelse(traindata$Q117193=="Standard hours",1,ifelse(traindata$Q117193=="Odd hours",-1,traindata$Q117193))
traindata$Q118232 = ifelse(traindata$Q118232=="Idealist",1,ifelse(traindata$Q118232=="Pragmatist",-1,traindata$Q118232))
traindata$Q119650 = ifelse(traindata$Q119650=="Giving",1,ifelse(traindata$Q119650=="Receiving",-1,traindata$Q119650))
traindata$Q120194 = ifelse(traindata$Q120194=="Study first",1,ifelse(traindata$Q120194=="Try first",-1,traindata$Q120194))
traindata$Q120472 = ifelse(traindata$Q120472=="Science",1,ifelse(traindata$Q120472=="Art",-1,traindata$Q120472))
traindata$Q122771 = ifelse(traindata$Q122771=="Public",1,ifelse(traindata$Q122771=="Private",-1,traindata$Q122771))

Factind = sapply(traindata, is.character)
traindata[Factind] = lapply(traindata[Factind], factor)


##
drops = c( 'UserID' )
finaldata = traindata[, !( names( traindata ) %in% drops )]

# clean up YOB---------------------------------
#finaldata$YOB[finaldata$YOB < 1930] = 0
#Set the NA value as the average age
finaldata$YOB[finaldata$YOB > 2014] = 0
finaldata$YOB[is.na(finaldata$YOB)] = 1979


summary(finaldata$EducationLevel)
levels(finaldata$EducationLevel) <- c("Bachelor's Degree", "Bachelor's Degree", "Current K-12", "Current Undergraduate", "Master's Degree", "Current K-12", "Master's Degree")
#This next step is very important as it removes the merged levels from the contrast matrix. Contrast matrix is the matrix that shows the encoding of the factors
finaldata$EducationLevel = factor(finaldata$EducationLevel)
summary(finaldata$EducationLevel)

summary(finaldata$Income)
levels(finaldata$Income) <- c("$100,001 - $150,000", "$25,001 - $50,000", "$50,000 - $74,999", "$50,000 - $74,999", "over $150,000", "$25,001 - $50,000")
finaldata$Income = factor(finaldata$Income)
summary(finaldata$Income)

summary(finaldata$HouseholdStatus)
levels(finaldata$HouseholdStatus) <- c("Domestic Partners (no kids)","Domestic Partners (w/kids)","Married (no kids)","Married (no kids)","Single (no kids)","Single (w/kids)")
finaldata$HouseholdStatus = factor(finaldata$HouseholdStatus)
summary(finaldata$HouseholdStatus)


summary(finaldata$Party)
levels(finaldata$Party) <- c("Democrat","Democrat","Libertarian","Democrat","Republican")
finaldata$Party = factor(finaldata$Party)
summary(finaldata$Party)

#Using mice to fill in the NA for other variables
simple = finaldata[c("Gender", "Income", "HouseholdStatus", "EducationLevel", "Party")]
summary(simple)
set.seed(144)
imp = complete(mice(simple))
summary(imp)
finaldata$Gender <- imp$Gender
finaldata$Income <- imp$Income
finaldata$HouseholdStatus <- imp$HouseholdStatus
finaldata$EducationLevel <- imp$EducationLevel
finaldata$Party <- imp$Party





#train / val split---------------------------------------
#for all the models we train on 'train' and validate on 'val'
#for models that do 10-fold cross validation, we use finaldata
set.seed(144)
split = sample.split(finaldata$Happy, SplitRatio=0.7)
train = subset(finaldata, split==TRUE)
val = subset(finaldata, split==FALSE)



#Logistic regression--------------------------------------------
logitmodel = glm(Happy~ Gender + HouseholdStatus + EducationLevel + Income + Q102687 + Q98059 +  Q118237 + Q101162 + Q107869 + Q102289 + Q102906 + YOB + Party + Q120014 + Q106997 + Q119334 + Q115610 +Q98869 + Q108855 + votes + Q108342 + Q98197 + Q116953+ Q108343+ Q121011 + Q111848 + Q117186 + Q113181+ Q108856 + Q102089 + Q114961 + Q116197 + Q115390 + Q99716 + Q117193+ Q123621+ Q106389+ Q115611+ Q124122+ Q120650+ Q118233+ Q109367+ Q115777+ Q116441+ Q122120+ Q124742+ Q102674+ Q116448+ Q101163+ Q119650+ Q103293+ Q98578 + Q100562, data=train, family=binomial)
summary(logitmodel)
threshold=0.5
predictLogit = predict(logitmodel, type="response", newdata=val)
table(val$Happy, as.numeric(predictLogit >= threshold))
predLogit= prediction(predictLogit, val$Happy)
logitauc <- as.numeric(performance(predLogit, "auc")@y.values)
cat( "Logistic Regression AUC:", logitauc, "\n" )


tr.control = trainControl(method = "cv", number = 10)
logitcv = train(Happy ~ Gender + HouseholdStatus + EducationLevel + Income + Q102687 + Q98059 +  Q118237 + Q101162 + Q107869 + Q102289 + Q102906 + YOB + Party + Q120014 + Q106997 + Q119334 + Q115610 +Q98869 + Q108855 + votes + Q108342 + Q98197 + Q116953+ Q108343+ Q121011 + Q111848 + Q117186 + Q113181+ Q108856 + Q102089 + Q114961 + Q116197 + Q115390 + Q99716 + Q117193+ Q123621+ Q106389+ Q115611+ Q124122+ Q120650+ Q118233+ Q109367+ Q115777+ Q116441+ Q122120+ Q124742+ Q102674+ Q116448+ Q101163+ Q119650+ Q103293+ Q98578 + Q100562, data = finaldata, method = "glm", trControl = tr.control, family=binomial)
logitcv
logitcv$finalModel$family



#svm-linear
svmmodel <- svm(Happy~., data=finaldata, kernel='linear',cost=100)
summary(svmmodel)
svmmodelacc <- svm(Happy~., data=finaldata, kernel='linear',cost=100,cross=10)

predsvm <- predict(svmmodel, newdata=val)
table(val$Happy, as.numeric(predsvm>=0))
predSVM= prediction(predsvm, val$Happy)
svmauc <- as.numeric(performance(predSVM, "auc")@y.values)
cat( "SVM Linear AUC:", svmauc, "\n" )



svmlincv <- tune.svm(Happy~., data = finaldata, cost = 10^(0:2))
summary(svmlincv)

#svm-gaussian
svmmodel2 <- svm(Happy~., data=finaldata, kernel='radial', gamma=0.5, cost=100)
summary(svmmodel2)

predsvm <- predict(svmmodel2, newdata=val)
table(val$Happy, as.numeric(predsvm>=0))
predSVM= prediction(predsvm, val$Happy)
svmauc2 <- as.numeric(performance(predSVM, "auc")@y.values)
cat( "SVM Gaussian AUC:", svmauc2, "\n" )

svmcv <- tune.svm(Happy~., data = finaldata, gamma = 10^(-6:-1), cost = 10^(0:2))
summary(svmcv)



#CART trees


# random forest

y_val = as.factor( val$Happy )
ntree = 100
set.seed(144)
rfmodel = randomForest( Happy ~ .-Q122769, data = train, ntree = ntree, nodesize=30 )
rfmodel = randomForest( Happy ~ Gender + Q118237 + Q101162 + Q107869 + HouseholdStatus + EducationLevel + Income + Q102289 + Q102906 + YOB + Party + Q120014 + Q106997 + Q119334 + Q115610 +Q98869 + Q108855 + votes + Q108342 + Q98197 + Q116953+ Q108343+ Q121011 + Q111848 + Q117186 + Q113181+ Q108856 + Q102089 + Q114961 + Q116197 + Q115390 + Q99716 + Q117193+ Q123621+ Q106389+ Q115611+ Q124122+ Q120650+ Q118233+ Q109367+ Q115777+ Q116441+ Q122120+ Q124742+ Q102674+ Q116448+ Q101163+ Q119650+ Q103293+ Q98578 + Q100562 , data = train, ntree = ntree, nodesize=30 )
p <- predict( rfmodel, val, type = 'prob' )
probs =  p[,2]

auc = colAUC( probs, y_val )
auc = auc[1]
cat( "Random forest AUC:", auc )

varImpPlot( rfmodel, n.var = 20)


# naive bayes

nb = naiveBayes( Happy ~ ., data = train )

# for predicting
drops = c( 'Happy' )
x_val = val[, !( names( val ) %in% drops )]

p = predict( nb, x_val, type = 'raw' )
probs =  p[,2]

auc = colAUC( probs, y_val )
auc = auc[1]

cat( "\n\n" )
cat( "Naive Bayes AUC:", auc, "\n" )



#Submission using testdataset

testdata = read.csv ("test.csv", na.strings=c("","NA"), stringsAsFactors=FALSE)

#Scaling down values
testdata[,8:108][testdata[,8:108] == "Yes"] = 1
testdata[,8:108][testdata[,8:108] == "No"] = -1
testdata[,8:108][is.na(testdata[,8:108])] = 0

testdata$Q98059 = ifelse(testdata$Q98059=="Yes",1,ifelse(testdata$Q98059=="Only-child",-1,testdata$Q98059))
#Or
#testdata$Science_or_Art[testdata$Science_or_Art=="Art"]=1;
#testdata$Science_or_Art[testdata$Science_or_Art=="Science"]=-1;
testdata$Q99982 = ifelse(testdata$Q99982=="Check!",1,ifelse(testdata$Q99982=="Nope",-1,testdata$Q99982))
testdata$Q101162 = ifelse(testdata$Q101162=="Optimist",1,ifelse(testdata$Q101162=="Pessimist",-1,testdata$Q101162))
testdata$Q101163 = ifelse(testdata$Q101163=="Dad",1,ifelse(testdata$Q101163=="Mom",-1,testdata$Q101163))
testdata$Q102089 = ifelse(testdata$Q102089=="Rent",1,ifelse(testdata$Q102089=="Own",-1,testdata$Q102089))
testdata$Q106997 = ifelse(testdata$Q106997=="Yay people!",1,ifelse(testdata$Q106997=="Grrr people",-1,testdata$Q106997))
testdata$Q108342 = ifelse(testdata$Q108342=="Online",1,ifelse(testdata$Q108342=="In-person",-1,testdata$Q108342))
testdata$Q108855 = ifelse(testdata$Q108855=="Yes!",1,ifelse(testdata$Q108855=="Umm...",-1,testdata$Q108855))
testdata$Q108856 = ifelse(testdata$Q108856=="Socialize",1,ifelse(testdata$Q108856=="Space",-1,testdata$Q108856))
testdata$Q108950 = ifelse(testdata$Q108950=="Cautious",1,ifelse(testdata$Q108950=="Risk-friendly",-1,testdata$Q108950))
testdata$Q110740 = ifelse(testdata$Q110740=="Mac",1,ifelse(testdata$Q110740=="PC",-1,testdata$Q110740))
testdata$Q111580 = ifelse(testdata$Q111580=="Supportive",1,ifelse(testdata$Q111580=="Demanding",-1,testdata$Q111580))
testdata$Q113583 = ifelse(testdata$Q113583=="Tunes",1,ifelse(testdata$Q113583=="Talk",-1,testdata$Q113583))
testdata$Q113584 = ifelse(testdata$Q113584=="People",1,ifelse(testdata$Q113584=="Technology",-1,testdata$Q113584))
testdata$Q114386 = ifelse(testdata$Q114386=="TMI",1,ifelse(testdata$Q114386=="Mysterious",-1,testdata$Q114386))
testdata$Q115777 = ifelse(testdata$Q115777=="Start",1,ifelse(testdata$Q115777=="End",-1,testdata$Q115777))
testdata$Q115899 = ifelse(testdata$Q115899=="Circumstances",1,ifelse(testdata$Q115899=="Me",-1,testdata$Q115899))
testdata$Q116197 = ifelse(testdata$Q116197=="A.M.",1,ifelse(testdata$Q116197=="P.M.",-1,testdata$Q116197))

testdata$Q116881 = ifelse(testdata$Q116881=="Happy",1,ifelse(testdata$Q116881=="Right",-1,testdata$Q116881))
testdata$Q117186 = ifelse(testdata$Q117186=="Hot headed",1,ifelse(testdata$Q117186=="Cool headed",-1,testdata$Q117186))
testdata$Q117193 = ifelse(testdata$Q117193=="Standard hours",1,ifelse(testdata$Q117193=="Odd hours",-1,testdata$Q117193))
testdata$Q118232 = ifelse(testdata$Q118232=="Idealist",1,ifelse(testdata$Q118232=="Pragmatist",-1,testdata$Q118232))
testdata$Q119650 = ifelse(testdata$Q119650=="Giving",1,ifelse(testdata$Q119650=="Receiving",-1,testdata$Q119650))
testdata$Q120194 = ifelse(testdata$Q120194=="Study first",1,ifelse(testdata$Q120194=="Try first",-1,testdata$Q120194))
testdata$Q120472 = ifelse(testdata$Q120472=="Science",1,ifelse(testdata$Q120472=="Art",-1,testdata$Q120472))
testdata$Q122771 = ifelse(testdata$Q122771=="Public",1,ifelse(testdata$Q122771=="Private",-1,testdata$Q122771))

Factind = sapply(testdata, is.character)
testdata[Factind] = lapply(testdata[Factind], factor)

##

# clean up YOB---------------------------------
#testdata$YOB[testdata$YOB < 1930] = 0
#Set the NA value as the average age
#testdata$YOB[testdata$YOB > 2014] = 0
testdata$YOB[is.na(testdata$YOB)] = 1979

summary(testdata$EducationLevel)
levels(testdata$EducationLevel) <- c("Bachelor's Degree", "Bachelor's Degree", "Current K-12", "Current Undergraduate", "Master's Degree", "Current K-12", "Master's Degree")
#This next step is very important as it removes the merged levels from the contrast matrix. Contrast matrix is the matrix that shows the encoding of the factors
testdata$EducationLevel = factor(testdata$EducationLevel)
summary(testdata$EducationLevel)

summary(testdata$Income)
levels(testdata$Income) <- c("$100,001 - $150,000", "$25,001 - $50,000", "$50,000 - $74,999", "$50,000 - $74,999", "over $150,000", "$25,001 - $50,000")
testdata$Income = factor(testdata$Income)
summary(testdata$Income)

summary(testdata$HouseholdStatus)
levels(testdata$HouseholdStatus) <- c("Domestic Partners (no kids)","Domestic Partners (w/kids)","Married (no kids)","Married (no kids)","Single (no kids)","Single (w/kids)")
testdata$HouseholdStatus = factor(testdata$HouseholdStatus)
summary(testdata$HouseholdStatus)

summary(testdata$Party)
levels(testdata$Party) <- c("Democrat","Democrat","Libertarian","Democrat","Republican")
testdata$Party = factor(testdata$Party)
summary(testdata$Party)

#Using mice to fill in the NA for other variables
simple = testdata[c("Gender", "Income", "HouseholdStatus", "EducationLevel", "Party")]
summary(simple)
set.seed(144)
imptest = complete(mice(simple))
summary(imptest)
testdata$Gender <- imptest$Gender
testdata$Income <- imptest$Income
testdata$HouseholdStatus <- imptest$HouseholdStatus
testdata$EducationLevel <- imptest$EducationLevel
testdata$Party <- imptest$Party

logittestmodel = glm(Happy~ Gender + HouseholdStatus + EducationLevel + Income + Q102687 + Q98059 +  Q118237 + Q101162 + Q107869 + Q102289 + Q102906 + YOB + Party + Q120014 + Q106997 + Q119334 + Q115610 +Q98869 + Q108855 + votes + Q108342 + Q98197 + Q116953+ Q108343+ Q121011 + Q111848 + Q117186 + Q113181+ Q108856 + Q102089 + Q114961 + Q116197 + Q115390 + Q99716 + Q117193+ Q123621+ Q106389+ Q115611+ Q124122+ Q120650+ Q118233+ Q109367+ Q115777+ Q116441+ Q122120+ Q124742+ Q102674+ Q116448+ Q101163+ Q119650+ Q103293+ Q98578 + Q100562, data=train, family=binomial)
summary(logittestmodel)
threshold=0.5
Probability1 = predict(logittestmodel, type="response", newdata=testdata)


#ptest <- predict( rfmodel, testdata, type = 'prob' )
#Probability1 =  ptest[,2]

submission <- data.frame(testdata$UserID, Probability1)
write.csv(submission, file="sub2.csv", row.names=FALSE)
