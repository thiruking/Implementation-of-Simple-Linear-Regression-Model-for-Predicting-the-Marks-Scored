# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THIRUMALAI K
RegisterNumber: 212224240176 
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```
## Output:
## Head Values
![image](https://github.com/user-attachments/assets/0a41d7c7-0851-4102-aa59-11e2c7281129)

## TAIL VALUES

![image](https://github.com/user-attachments/assets/c51be7d2-5240-4103-881c-e23b6c2f06b4)

## COMPARE DATASET
![image](https://github.com/user-attachments/assets/5bb4b940-86f5-4a2b-80a7-b8aa245ee53f)

## Predication values of X and Y
![image](https://github.com/user-attachments/assets/ef4e15b6-5852-4311-a693-a380f3180eac)

## Training set
![image](https://github.com/user-attachments/assets/b1d0a68b-d027-4818-b546-f6d8ec3baeab)

## Testing Set
![image](https://github.com/user-attachments/assets/d15c3f29-57b8-4ffe-a70a-2de2dcc1c0aa)

## MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/94cf5b4a-4f19-4df7-b1f8-f62c5685fa3a)
<br>
<br>

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
