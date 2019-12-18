# Решение системы линейных уравнений с трехдиагональной матрицей

iimport numpy as np

def sol3(A,b):

	mainDiagonal = np.array([A[i,i] for i in range(len(A))],float)
	upperSideDiagonal = np.array([A[i,i+1] for i in range(len(A)-1)],float)
	lowerSideDiagonal = np.array([A[i+1,i] for i in range(len(A)-1)],float)
	zerColmn1 = np.zeros(len(A),float)
	zerColmn1[1] = -upperSideDiagonal[0]/mainDiagonal[0]

	for i in range(2,len(A)):
		zerColmn1[i] = -upperSideDiagonal[i-1]/(mainDiagonal[i-1]+lowerSideDiagonal[i-1]*zerColmn1[i-1])

	zerColmn2 = np.zeros(len(A),float)
	zerColmn2[1] = b[0]/mainDiagonal[0]

	for i in range(2,len(A)):
		zerColmn2[i] = (-lowerSideDiagonal[i-1]*zerColmn2[i-1] + b[i-1]) / (mainDiagonal[i-1] + lowerSideDiagonal[i-1]*zerColmn1[i-1])

	res = np.zeros(len(A),float)
	res[-1] = (-lowerSideDiagonal[-1]*zerColmn2[-1] + b[-1])/(mainDiagonal[-1] + lowerSideDiagonal[-1]*zerColmn1[-1])

	for i in range(len(A)-2,-1,-1):
		res[i] = zerColmn1[i+1]*res[i+1] + zerColmn2[i+1]
	return res
    

A = np.array([[157,8,74,200],[7,233,6,12],[0,7,25,82],[0,4,33,268]])
b = np.array([21,35,56,8])
res = sol3(A,b)
result = np.dot(A,res)
print(result)