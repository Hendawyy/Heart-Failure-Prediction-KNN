# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:54:40 2021

Seif Hendawy, Shehab Gamal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score,recall_score,f1_score

#Read Data from Directory in Pc.
Dataset = pd.read_csv('D:/AAST/Year 3/Semester 6/Introduction to Artificial Intelligence (CS366 )/Project/heart_failure_clinical_records_dataset.csv')

#Check if There is Any Null Value in The Dataset.
print(Dataset.isnull().sum())

#Split Dataset to X, Y.
X = Dataset.iloc [: ,:12].values
Y = Dataset.iloc [: ,-1].values


#Using the Train_test_split to Split dataset into Train and test.
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Using Feature Scaling to Scale Values in X To Numerical Values Between -1.5 & 1.5.
Scaler = StandardScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train = Scaler.fit_transform(x_train)
x_test = Scaler.fit_transform(x_test)



#Optimal Value Of K
acc = []

for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(x_train,y_train)
    Z = neigh.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, Z))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

N = acc.index(max(acc));

error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(x_train,y_train)
 pred_i = knn.predict(x_test)
 error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

E = error_rate.index(min(error_rate))

#KNN with K = Optimum Using The Distance Equation Minkowski.
CM = KNeighborsClassifier(n_neighbors=N, metric ='minkowski')
CM.fit(x_train, y_train)
y_predM = CM.predict(x_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Minkowski.
kcfM = confusion_matrix(y_test, y_predM)
print('Confusion Matrix : ')
print(kcfM)

#Classification Report For K = Optimum  Using The Distance Equation Minkowski.
kcrM = classification_report(y_test, y_predM)
print('Classification Report : ')
print(kcrM)

#Accuracy Score For K = Optimum  Using The Distance Equation Minkowski.
kasM = accuracy_score(y_test, y_predM)
print('Accuracy :',kasM)

#Precision Score For K = Optimum  Using The Distance Equation Minkowski.
kpsM = precision_score(y_test , y_predM)
print("Precision:" , kpsM)

#Recall Score For K = Optimum  Using The Distance Equation Minkowski.
krsM = recall_score(y_test , y_predM)
print("Recall:" , krsM)

#F1 Score For K = Optimum  Using The Distance Equation Minkowski.
kfsM = f1_score(y_test , y_predM)
print("f1_score:" , kfsM)


#Intialize Data to be Put in The Dafarame. 
KDF = {'Confusion Matrix' : pd.Series([kcfM], index =['Minkowski']),
      'Accuracy Score' : pd.Series([kasM], index =['Minkowski']),
      'Precision Matrix' : pd.Series([kpsM], index =['Minkowski']),
      'Recall Score' : pd.Series([krsM], index =['Minkowski']),
      'F1 Score' : pd.Series([kfsM], index =['Minkowski'])} 

#Create an Comparison Dataframe That Holds a comparison between The Accuracy Scores and the confusion matrices. 
KOPT = pd.DataFrame(KDF) 

print('K = Optimum\n',KOPT)
