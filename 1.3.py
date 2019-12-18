#Разложение Холецкого

import numpy as np
from math import sqrt

def Hol(A):

	result = np.zeros((len(A),len(A)),float)

	for i in range(0,len(A),1):

		temp1 = 0
		temp2 = 0

		for j in range(0,i,1):
			for k in range(0,j,1):
				temp1 = temp1 + result[i,k]*result[j,k]
			result[i,j] = (A[i,j] - temp1)/result[j,j]
            
		for t in range(0,i,1):
			temp2 = temp2 + result[i,t]*result[i,t]
		result[i,i] = sqrt(A[i,i] - temp2)

	return result

def bottomtr(C,b):

	y = np.zeros(len(C))

	for i in range(0,len(C),1):

		temp = 0

		for j in range (0,i,1):
			temp = temp + C[i,j]*y[j]
		y[i] = (b[i] - temp)/C[i,i]

	return(y)

def toptr(CT,y):

	x = np.zeros(len(CT),float)

	for i in range (len(CT),0,-1):

		temp = 0

		for j in range (i,len(CT),1):
			temp = temp + CT[i-1,j]*x[j]
		x[i-1] = (y[i-1] - temp)/CT[i-1,i-1]

	return(x)


A = np.array([[15,1,8],[5,13,-1],[11,-1,17]],float)
x = np.array([7,4,5])
b = np.dot(A,x)
C = Hol(A)
print(C)
y = bottomtr(C,b)
x = toptr(C.transpose(),y)
print(x)