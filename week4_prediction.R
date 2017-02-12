
## @knitr setLocation
library(caret)
library(randomForest)

train_data_loc<-"/Users/rahullagarwal/Code/JHU/git/practicalmachinelearning/pml-training.csv"
test_data_loc<-"/Users/rahullagarwal/Code/JHU/git/practicalmachinelearning/pml-testing.csv"
final_output_loc<-"/Users/rahullagarwal/Code/JHU/git/practicalmachinelearning/week4_test_predicted.csv"
set.seed(100)

## @knitr dataPrep1

data=read.csv(train_data_loc)
testing_final=read.csv(test_data_loc)
inTrain=createDataPartition(y=data$classe ,p=0.7, list=FALSE)
training=data[inTrain,]
validation=data[-inTrain,]
vars=c('user_name','roll_belt','pitch_belt','yaw_belt','total_accel_belt','gyros_belt_x','gyros_belt_y','gyros_belt_z','accel_belt_x','accel_belt_y','accel_belt_z','magnet_belt_x','magnet_belt_y','magnet_belt_z','roll_arm','pitch_arm','yaw_arm','total_accel_arm','gyros_arm_x','gyros_arm_y','gyros_arm_z','accel_arm_x','accel_arm_y','accel_arm_z','magnet_arm_x','magnet_arm_y','magnet_arm_z','roll_dumbbell','pitch_dumbbell','yaw_dumbbell','total_accel_dumbbell','gyros_dumbbell_x','gyros_dumbbell_y','gyros_dumbbell_z','accel_dumbbell_x','accel_dumbbell_y','accel_dumbbell_z','magnet_dumbbell_x','magnet_dumbbell_y','magnet_dumbbell_z','roll_forearm','pitch_forearm','yaw_forearm','total_accel_forearm','gyros_forearm_x','gyros_forearm_y','gyros_forearm_z','accel_forearm_x','accel_forearm_y','accel_forearm_z','magnet_forearm_x','magnet_forearm_y','magnet_forearm_z','classe')
training=training[,vars]
subset_continious=subset(training, select = -c(classe,user_name) )
qplot(training$yaw_belt,training$total_accel_belt , color=training$user_name)

## @knitr pca
pca_dimensions=18
pca=prcomp(subset_continious , center=T , scale=T)
features_after_pca=pca$x[,1:pca_dimensions]
features_after_pca=cbind(features_after_pca , subset(training, select = c(classe,user_name)))
validation_pca<-predict(pca, newdata=validation)
validation_features_after_pca=cbind(validation_pca[,1:pca_dimensions] , subset(validation, select = c(classe,user_name)))
summary(pca)

## @knitr randomForest
number_of_tree=200
accuracy=function(values,prediction){ sum(ifelse(values==prediction,1,0))/length(values) } 
modRF = randomForest(classe~., features_after_pca, ntree=number_of_tree , do.trace=FALSE)
pred=predict(modRF,validation_features_after_pca)
accuracy(validation$classe , pred )
plot(modRF)

## @knitr testPrediction
test_pca<-predict(pca, newdata=testing_final)
test_features_after_pca=cbind(test_pca[,1:pca_dimensions] , subset(testing_final, select = c(user_name)))
predTest=predict(modRF,test_features_after_pca)
results<-cbind(testing_final,data.frame(classe=predTest)) 
write.csv(results, file = final_output_loc) 

## @knitr compareLDA
modLDA=train(classe~. , method="lda" , data=training, prox=TRUE)
predLDA=predict(modLDA,validation)
accuracy(validation$classe , predLDA )


