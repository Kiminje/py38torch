import numpy as np
import numpy.linalg as lin
#1.
A = np.array([[-5, -6, 6], [-9, -8, 12], [-12, -12, 16]])
A_hat = np.array([[-5, -15/2, -3], [-15/2, -8, 0], [3, 0, 16]])
x = np.array([[1, 1/2, 1], [1, -1, 0], [0, 1, 1]])
x = np.transpose(x)
print(x)
# x_inv = np.array([[2, 2, 0], [-1, -2, 0], [-2, -2, 0]])
x_inv = np.array([[2, 2, -2], [-1, -2, 2], [-2, -2, 3]])
print("x_inv\n", x_inv)
print(np.dot(x, x_inv))
print("inverse matrix of X\n", x_inv)
#
print("diagonalize\n", np.dot(x_inv, np.dot(A, x)))
#
# print(lin.eigvals(A), "\n", np.di)
w, v = lin.eig(A)
print("w\n", w)
v = v*3
print("v\n", v)
v_inv = lin.inv(v)
print("v_inv\n", v_inv)
print("x-1 A x = D\n", np.dot(v_inv, np.dot(A, v)))
