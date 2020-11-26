# height (cm)
#X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
#y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data
import numpy as np
import matplotlib.pyplot as plt
import csv
x = []
y1 = []
with open('linear_regression_file.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:  # row là một ma trận có hai cột x,y
        x.append(row[0])  # thêm row[0](x) và mảng x
        y1.append(row[1])
        print(y1)
    X = np.array([x],dtype=int).T   
    y= np.array([y1],dtype=int).T 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
print(Xbar,"\n",Xbar.T,"\n",A)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0
y_ = w_0 + w_1*160
print('Can nang 160cm',y_)
# Drawing the fitting line
plt.plot(X.T, y.T, 'ro')     # data
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()