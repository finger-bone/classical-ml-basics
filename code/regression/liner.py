import numpy as np

def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)
    xx = np.einsum('ij,ik->jk', x, x)
    xy = np.einsum('ij,i->j', x, y)
    x = np.einsum('ij->j', x)
    y = np.einsum('i->', y)
    w = np.linalg.inv(xx - np.outer(x, x)) @ (xy - x * y)
    b = y - w @ x
    return w, b

def predict(x, w, b):
    x = np.array(x)
    return x @ w + b

train_x = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
train_y = [3, 5, 7, 9, 11]
w, b = linear_regression(train_x, train_y)
print(w, b)
print(predict([[6, 7], [7, 8]], w, b))
