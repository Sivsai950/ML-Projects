
from sklearn.datasets import load_breast_cancer, fetch_california_housing #instead of load_bostod, import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression,  LinearRegression
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score,f1_score,confusion_matrix,roc_auc_score,
    mean_squared_error,r2_score
)
import numpy as np
import pandas as pd

 #Load the data
data=load_breast_cancer()
X=data.data
Y=data.target #1=benigs,0=maliagnant

#split into training ans test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
 
# train a sample logistic regression model
model=LogisticRegression(max_iter=10000)
model.fit(X_train,Y_train)
#predict on test data
Y_pred=model.predict(X_test)
Y_proba=model.predict_proba(X_test)[:,1]


print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("precision:",precision_score(Y_test, Y_pred))
print("recall:",recall_score(Y_test, Y_pred))
print("f1 score:",f1_score(Y_test, Y_pred))
print("confusion matrix:\n",confusion_matrix(Y_test, Y_pred))
print("ROC-AUC score:",roc_auc_score(Y_test, Y_proba))


