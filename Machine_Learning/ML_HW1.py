import numpy as np

from scipy.linalg import null_space

A = np.array([[1, 1], [1, 1]])
ns = null_space(A)
from numpy.linalg import matrix_rank

matrix_rank(np.array([[1, 1], [1, 1]]))

from sympy import *

M = Matrix([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]])

print("Matrix : {} ".format(M))

# Use sympy.rref() method
M_rref = M.rref()

print("The Row echelon form of matrix M and the pivot columns : {}".format(M_rref))