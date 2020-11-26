import numpy as np
import matplotlib.pyplot as plt
import csv
def prod(w,X):
    return np.dot(w.T, X)
def sigmoid(s):
    return 1/(1 + np.exp(-s))
def my_logistic_sigmoid_regression(X, y, w_init, eta, epsilon = 1e-3, M = 10000):
    w = [w_init]
    N = X.shape[1]              #so cot = 2
    d = X.shape[0]                #số ptu input=so hang   (X: Xbar)         
    count = 0
    check_w_after = 20
    while count < M:
        # mix data
        mix_id = np.random.permutation(N)  #mix_id: mang N ptu cac so ngau nhien tu 0 den N-1
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)     
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))        
            w_new = w[-1] + eta*(yi - zi)*xi       #cap nhat lai w
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < epsilon:
                    return w
            w.append(w_new)
    return w
def display(w,X,y):
    X0 = X[1, np.where(y == 0)][0]
    y0 = y[np.where(y == 0)]
    X1 = X[1, np.where(y == 1)][0]
    y1 = y[np.where(y == 1)]
    plt.plot(X0, y0, 'ro', markersize=8)
    plt.plot(X1, y1, 'bs', markersize=8)
    xx = np.linspace(0, 6, 1000)
    w0 = w[-1][0][0]
    w1 = w[-1][1][0]
    threshold = -w0 / w1
    yy = sigmoid(w0+w1*xx)
    plt.axis([-2, 8, -1, 2])
    plt.plot(xx, yy, 'g-', linewidth=2)
    plt.plot(threshold, .5, 'y^', markersize=8)
    plt.xlabel('studying hours')
    plt.ylabel('predicted probability of pass')
    plt.show()
if __name__ == '__main__':
    print("ok")
    # data:
    X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
    # #print(X)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
#     x = []
#     y1 = []
# with open('file.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:  # row là một ma trận có hai cột x,y
#         x.append(row[0])  # thêm row[0](x) và mảng x
#         y1.append(row[1])
#     X = np.array([x],dtype=float).T   
#     y= np.array(y1,dtype=int).T 
#     print("x: ",X)
#     print("y: ",y)
    # extended data
    Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    print("Xbar:")
    print(Xbar)
    eta = .05
    d = Xbar.shape[0]
    print("d:")
    print(d)
    w_init = np.random.randn(d, 1)
    print("w_init")
    print(w_init)
    w = my_logistic_sigmoid_regression(Xbar, y, w_init, eta)
    display(w,Xbar,y)
