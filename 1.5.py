import numpy as np

def sd(A, b, eps):
    x = np.array([0] * len(b))
    U = np.triu(A, k = 1)
    L = np.tril(A, k = -1)
    D = np.diagflat(np.diag(A))
    
    i = 0
    while True:
        y = np.dot(np.linalg.inv(L + D), np.dot(-U, x) + b)
        if (np.linalg.norm(x - y) < eps):
            return y
        x = y
        
        i += 1
        if (i > 200):
            return x
            
A = np.array([[0.24, 0.3, 1], [0.27, 0.07, 0.15], [0.05, 0.05, 1]])
b = np.array([1, 1, 1])
x = sd(A, b, 1e-9)
print("solution:", x)
print("check: ||A x - b|| =", np.linalg.norm(np.dot(A, x) - b))