# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import essential libraries for data manipulation, numerical operations, plotting, and regression analysis.
2. Load and Explore Data: Load a CSV dataset using pandas, then display initial and final rows to quickly explore the data's structure.
3. Prepare and Split Data: Divide the data into predictors (x) and target (y). Use train_test_split to create training and testing subsets for model building and evaluation.
4. Train Linear Regression Model: Initialize and train a Linear Regression model using the training data. Visualize and Evaluate: Create scatter plots to visualize data and regression lines for training and testing. Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to quantify model performance.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kancharla Narmadha
RegisterNumber:  212222110016
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("student_scores.csv") 
df.head()
df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="orangered",s=60)
plt.plot(x_train,regressor.predict(x_train),color="darkviolet",linewidth=4)
plt.title("hours vs scores(training set)",fontsize=24)
plt.xlabel("Hours",fontsize=18)
plt.ylabel("scores",fontsize=18)
plt.show()

plt.scatter(x_test,y_test,color="seagreen",s=60)
plt.plot(x_test,regressor.predict(x_test),color="cyan",linewidth=4)
plt.title("hours vs scores(training set)",fontsize=24)
plt.xlabel("Hours",fontsize=18)
plt.ylabel("scores",fontsize=18)
plt.show()


mse=mean_squared_error(_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
### Head:
![image](https://github.com/kancharlaNarmadha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559316/7f7d0f72-9836-4c6a-b6f9-ec20f503d483)

### Tail:
![image](https://github.com/kancharlaNarmadha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559316/b183ca88-0be7-44e3-bab9-72ae40ac6af7)

### Array value of X:
![image](https://github.com/kancharlaNarmadha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559316/a3f1083d-7bd5-482b-b084-950dfe710ba3)

### Array value of Y:
![image](https://github.com/kancharlaNarmadha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559316/d4330e9a-ead2-417a-9dc5-8adf8a76565e)

### Values of Y prediction:
![image](https://github.com/kancharlaNarmadha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559316/199da14f-1a31-4211-9fb4-a968dd1cc3cd)


### Array values of Y test:
![image](https://github.com/kancharlaNarmadha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559316/2ac06125-c95f-459a-8618-cf1e38834c2b)

### Training Set Graph:
![image](https://github.com/kancharlaNarmadha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559316/ce3d859f-d81c-4c3d-a8fc-144ff4e969da)

### Test Set Graph:
![image](https://github.com/kancharlaNarmadha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559316/1ec7f023-7bf9-4b2d-9ac3-7431b9372990)

### Values of MSE, MAE and RMSE:
![image](https://github.com/kancharlaNarmadha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559316/55b440bf-cbe3-4da0-affd-489207f81255)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
