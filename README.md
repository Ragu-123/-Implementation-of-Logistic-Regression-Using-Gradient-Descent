# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. 
6.Obtain the graph.
 ```

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:RAGUNATH R 
RegisterNumber:212222240081
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data2.txt",delimiter = ',')
x= data[:,[0,1]]
y= data[:,2]
print('Array Value of x:')
x[:5]

print('Array Value of y:')
y[:5]

print('Exam 1-Score graph')
plt.figure()
plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label=' Not Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print('Sigmoid function graph: ')
plt.plot()
x_plot = np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()


def costFunction(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad = np.dot(x.T,h-y)/x.shape[0]
  return j,grad


print('X_train_grad_value: ')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
j,grad = costFunction(theta,x_train,y)
print(j)
print(grad)


print('y_train_grad_value: ')
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j


def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j

print('res.x:')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()

print('DecisionBoundary-graph for exam score: ')
plotDecisionBoundary(res.x,x,y)

print('Proability value: ')
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)


def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,x)==y)
*/
```

## Output:
## Array Value of x:
![image](https://github.com/Ragu-123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113915622/8b201df2-8da4-42d0-a0d2-af87c38e262d)
## Array Value of y:
![image](https://github.com/Ragu-123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113915622/c468e06d-b65e-4b39-aa2d-4a72ddd22d24)
## Exam 1 - score graph:
![image](https://github.com/Ragu-123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113915622/25c900fb-4d40-46e5-a25e-15c73d31b662)
## Sigmoid function graph:
![image](https://github.com/Ragu-123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113915622/1475a9b9-bbb0-401c-b50b-90b6ae88f225)
## X_train_grad value:
![image](https://github.com/Ragu-123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113915622/6b3d3292-0273-44bf-ac5a-0439f7595dfe)
## Y_train_grad value:
![image](https://github.com/Ragu-123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113915622/83f8a7be-603b-4b8d-8d42-b8a0822bfccc)
## Print res.x:
![image](https://github.com/Ragu-123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113915622/4d67b4e5-118a-4d44-a998-14aa0f4b8d24)
## Decision boundary - graph for exam score:
![image](https://github.com/Ragu-123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113915622/a93ea8b3-86df-4c5f-97e1-4faab5bf6d2c)
## Proability value:
![image](https://github.com/Ragu-123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113915622/8790b00c-77c4-49e8-b313-684d92e9fcca)
## Prediction value of mean:
![image](https://github.com/Ragu-123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113915622/898ccf3d-4c05-42c4-add6-6cf1c66946e6)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

