[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MqChnODK)

## Old school project **
### Common student methodology mistakes ###
    No separate handling of continuous and categorical features .Mean/median imputation is improper for categorical features, where mode imputation should be used 
    When doing grid search, some groups don't optimize the hyperparameters jointly but one by one. Also, hyperparameter search should be usually done on a logarithmic scale, not linear.
    The logistic loss function that is supposed to pass the test is designed for class labels {0,1}, while in the project, the class labels are in {-1,+1}. Many students directly applied the same loss function on the provided class labels without mapping them to the correct range.
    It seems like for the given dataset, there was a clear tradeoff between accuracy and F1-score: if you train too long, your F1-score deteriorates in favor of accuracy. Thus, it's important to monitor the target metric (in this case, F1-score) throughout the training and not only at the end.
    Conclusions drawn from the accuracy metric are illogical. Accuracy is not useful when you have an imbalanced dataset (i.e. you can have 1% positive samples and 99% negative, and  a model that always predicts negative will have a 99% accuracy). Use F-1 score or AUC-ROC instead.

ReadMe for the code run: 
All the functions used in the run code are from the implementations.py file
 1. the code does data cleaning and data processing by removing some useless features as phone or month of response. Then it fills the NaN values with 0 and Normalized and standardized the datas using the normalize_and_fill function. 
2. The dataset is divided between train part (80%) and test part. Then it does some data augmentation, you can choose the number of time you want to copy the sample with label +1. 
3. A ridge regression is applied on the new samples and using K-fold ethod, a optimization is made on the regularization parameter lambda. 
A best Penality Check is implemented: once the best lambda found and the optima weights w are also computed usning ridge, it is time to optimized the penalty that is used on the y vector to predict futur CVD.
4. A ridge regression is applied this time using reduced sample. A best penalty check is also implemented
5. Reg Logistic and SVM with gradient are then implemented, and still with a penalty check after, still using reduced samples because it is faster.
6. BONUS: a new ridge with polynomial features was done after the kernel lecture this wendnesday, surprisingly it gave us our better prediction, but we only did it with the reduced data. Doing it with the augmented data might have improved our results.
