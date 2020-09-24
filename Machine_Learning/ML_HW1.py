import numpy as np

from scipy.linalg import null_space

A = np.array([[1, 1], [1, 1]])
ns = null_space(A)
from numpy.linalg import matrix_rank

matrix_rank(np.array([[1, 1], [1, 1]]))

from sympy import *


def FindRrefNull(array):
    Mat = Matrix(array)
    Null = np.array(array)
    Null = null_space(Null)
    Mat_rref = Mat.rref()
    return Mat_rref, Null


#1
M1 = [[-2, 6, 8], [1/2, -2, 0], [0, 1/2, -2]]
print("\nQ1\n")

# Use sympy.rref() method
M1_rref, M1_null = FindRrefNull(M1)

print("The Row echelon form of matrix M and the pivot columns : {}".format(M1_rref))
print(M1_null)

#2
print("\nQ2\n")
M2 = [[0, 6, 8], [1/2, 0, 0], [0, 1/2, 0]]

# Use sympy.rref() method
M2_rref, M2_null = FindRrefNull(M2)

print("The Row echelon form of matrix M and the pivot columns : {}".format(M2_rref))
print(M2_null)

#3
print("\nQ3\n")
M3 = [[1, 2, -3], [2, 4, -6], [-1, -2, 3]]

# Use sympy.rref() method
M3_rref, M3_null = FindRrefNull(M3)

print("The Row echelon form of matrix M and the pivot columns : {}".format(M3_rref))
print(M3_null)

#4
print("\nQ4\n")
M4 = [[3/2, 0, 3], [-3/2, 0, -3], [-3/2, 0, -3]]

# Use sympy.rref() method
M4_rref, M4_null = FindRrefNull(M4)

print("The Row echelon form of matrix M and the pivot columns : {}".format(M4_rref))
print(M4_null)

#5
print("\nQ5\n")
M5 = [[1, 0, -8, -7], [0, 1, 4, 3], [0, 0, 0, 0]]

# Use sympy.rref() method
M5_rref, M5_null = FindRrefNull(M5)

print("The Row echelon form of matrix M and the pivot columns : {}".format(M5_rref))
print(M5_null)