#models:

#1-LinearRegression: one of  the simple model that finds the liner relationship between the input and the output.
#Code: from sklearn.linear_model import LinearRegression and
# from sklearn.model_selection import train_test_split.
#important parameter:fit_intercept:it specifies whether the model should have bias or not.
#effect on overfitting & underfitting:it is a simple model, so it usually has a low risk of overfitting.


#2-SGDRegressor: one of  regression  model in scikit-learn
#Code: from sklearn.linear_model import SGDRegressor.
#it is used for regression and is fast on large datasets.
#importatnt parameters P random_state  & max_iter.


#3-Ridge(),Lasso(),ElasticNet()

#Ridge Regression :it shrinks the coefficients, but it doesnt set any coefficient to zero.
#Code:from sklearn.linear_model import Ridge.
#its goal is to reduce overfitting.

#Lasso:it sets some coefficient exactly to zero (it goal:reduce overfitting)
#Code:from sklearn.linear_model import Lasso

#ElasticNet:it consist of Ridge and lasso,so it is better than using  Ridge or  lasso in some cases.
#Code:from sklearn.linear_model import ElasticNet
#important parameters for all three models : Alpha,it determines the Regularization strenght.
#Alpha> a simpler model with a higher risk of underfitting vc Alpha<a more complex model with a higher risk of overfitting
#we use these 3 models to provent overfitting.




#4-KNeighbourRegressor:it use the nearest data  points for prrediction(KNN)
#Code: from sklearn.neighbors import KNeighborrsRegressor.
#the model finds the nearest neighbours and predicts their mean as a anwser
#import parameter:n_neighbors
#import point:first we should scale  the data becuse KNN works base on distance.


#5-DecisionTreeRegressor:one of the Regression models used for predictings number.
#code: from sklearn.tree import DecitionTreeRegressor.
#Example: ask question like a tree and devides the data into smaller groups.
#important parameter: max_depth , min_samples_split, random_state.
#effect on overfitting and underfitting.
#why is this model beneficial??: it dosent need scaling and works well nonlinear relationships.
#red point: it can become overfit fast.


#6-RndomForestRegressor:one  of the  models used for predicting data.
#Code:from sklearn.ensemble import RandomForestRegressor
#it combines several decition trees to creat s Random forest.
#gold point:lowest overfittiung and more accurate.
#import parameters:n_estimator , maz_depth , random_state.



#7-SVR:it find a line that keeps most data points within a certain range around it.sensetive about data sacl.
#Code:from sklearn.svm import SVR
#important parameter:C(low,high) & Kernel,
#Effect : C< the model become simpler ,underfitting risk increases vc C>the model become more complex and overfitting increases.


#8-LogesticResgression:one of the model classification regression (not regression)
#Code:from sklearn.linear_model import LogisticRegression
#important parameter:C :C>(Regularization lower,complex model), c<(Regularization higher ,simpeler model)


#9-SGDClassifier: one of Classification models in  scikit_learn.
#important parameter: Loss, max_iter ,learning_rate, random_state.
#Code: from sklearn.linear_model import SGDClassifier.
#Effec: Alpha> the model becomes simplers and overfitting decreases and risk of underfitting increased vc
#Alpha< the model become more complex and the risk of overfitting increases.
#importanat parameter: Loss, max_iter, learning_rate, alpha, random_state.


#10-KNeighbourClassifier: used for classification (output:Class)
#Code: from sklearn.neighbors import KNeighborsClassifier
#import parameters: n_neighbors, K, Weights, metric, algorithm.
#effect on overfitting and underfitting.


#11-DecisionTreeClassifier: one of the classification model(outpot:Class)
#Code: from sklearn.tree import DecitionTreeClassifier
#importanat parameters: max_depth, min_samples_aplit.
#Effect on :max_depth



#12-RandomForestClassifier(classification):combines several Decision trees.
#Code: from sklearn.ensemble import RandomForestClassifier.
#important parameters: n_estimators, max_depth, max_features.
#Effect on overfitting and underfitting on :
#n_estimators(usually improve generalization and reduces variance).
#max_depth< the risk of underfitting increases.max_depth> the model becomes more complex and ,
#the risk of overfitting increases.


#13-SVC: the model finds the best decision boundary  that has the maximum  diatance from 2 groups(this distance is called the margin)
#Code: from sklearn,SVM import SVC.
#import parameters: C, Kernel(linear, rbf, poly), Gamma .