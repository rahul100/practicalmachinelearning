########### 1. Enironment & Libraries & File locations#########################
rm(list=ls())  ### Removing prior dataframes & other data
warnings()
library(caret)

train_data_loc<-"/data/analytics/rahul/JHU/pml-training.csv"
test_data_loc<-"/data/analytics/rahul/JHU/pml-testing.csv"
final_output_loc<-"/data/analytics/rahul/JHU/week4_test_predicted.csv"
set.seed(100)

############ 2. Variables #######################################
pca_dimensions=18
number_of_tree=200
accuracy=function(values,prediction){ sum(ifelse(values==prediction,1,0))/length(values) } 
############ 3. Data preparation ################################
data=read.csv(train_data_loc)
testing_final=read.csv(test_data_loc)
inTrain=createDataPartition(y=data$classe ,p=0.7, list=FALSE)
training=data[inTrain,]
validation=data[-inTrain,]
vars=c('user_name','roll_belt','pitch_belt','yaw_belt','total_accel_belt','gyros_belt_x','gyros_belt_y','gyros_belt_z','accel_belt_x','accel_belt_y','accel_belt_z','magnet_belt_x','magnet_belt_y','magnet_belt_z','roll_arm','pitch_arm','yaw_arm','total_accel_arm','gyros_arm_x','gyros_arm_y','gyros_arm_z','accel_arm_x','accel_arm_y','accel_arm_z','magnet_arm_x','magnet_arm_y','magnet_arm_z','roll_dumbbell','pitch_dumbbell','yaw_dumbbell','total_accel_dumbbell','gyros_dumbbell_x','gyros_dumbbell_y','gyros_dumbbell_z','accel_dumbbell_x','accel_dumbbell_y','accel_dumbbell_z','magnet_dumbbell_x','magnet_dumbbell_y','magnet_dumbbell_z','roll_forearm','pitch_forearm','yaw_forearm','total_accel_forearm','gyros_forearm_x','gyros_forearm_y','gyros_forearm_z','accel_forearm_x','accel_forearm_y','accel_forearm_z','magnet_forearm_x','magnet_forearm_y','magnet_forearm_z','classe')
training=training[,vars]
subset_continious=subset(training, select = -c(classe,user_name) )
############ 4. Calculating PCA & Center & Scale #####################
## Train
pca=prcomp(subset_continious , center=T , scale=T)
features_after_pca=pca$x[,1:pca_dimensions]
features_after_pca=cbind(features_after_pca , subset(training, select = c(classe,user_name)))
## Validation
validation_pca<-predict(pca, newdata=validation)
validation_features_after_pca=cbind(validation_pca[,1:pca_dimensions] , subset(validation, select = c(classe,user_name)))
############ 5. Random Forest Classification using ######################
modRF = randomForest(classe~., features_after_pca, ntree=number_of_tree , do.trace=TRUE)
plot(modRF)
pred=predict(modRF,validation_features_after_pca)
#qplot(training$yaw_belt,training$total_accel_belt , color=training$user_name)
accuracy(validation$classe , pred )


############ 6. Final Preidction on test set #####################
## Train
test_pca<-predict(pca, newdata=testing_final)
test_features_after_pca=cbind(test_pca[,1:pca_dimensions] , subset(testing_final, select = c(user_name)))
predTest=predict(modRF,test_features_after_pca)
head(testing_final)
results<-cbind(testing_final,data.frame(classe=predTest)) 
write.csv(results, file = final_output_loc) 

#### Comparison with other models ################
###################LDA Model-- Using Raw Features ###########################################
modLDA=train(classe~. , method="lda" , data=training, prox=TRUE)
summary(modLDA)
predLDA=predict(modLDA,validation)
#qplot(training$yaw_belt,training$total_accel_belt , color=training$user_name)
accuracy(validation$classe , predLDA )
###################GBM Model-- Using Raw Features ###########################################
ModGBM = gbm(classe~.,training, distribution="gaussian",shrinkage=0.1,interaction.depth = 3, bag.fraction = 0.5,train.fraction = 1.0, cv.folds = 10,verbose = TRUE)
summary(ModGBM) 
predGBM=predict(ModGBM,validation)
predGBM
accuracy(validation$classe , predGBM )
###################Other Exploration work###########################################
plot(pca , type='l')
biplot(pca)
summary(pca)
sub_after_pca<-predict(pca,sub)
dim(sub_after_pca)
plot(pca$x[,1] , pca$x[,3])





