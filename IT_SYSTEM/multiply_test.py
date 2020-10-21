import numpy as np
import random


m1 = np.random.rand(2, 3)
m2 = np.ones(m1.shape)
# print(m2.ctypes)
# print(m1.shape)
# print(m2)
# m2[:, 0] = 0
# print("this is m2\n", m2)
# print(m2[:, 0])
#
# m3 = np.random.random(m2.shape)
# print(m3)
# mul1 = m2.dot(m3.T)
# print(mul1)
# result = np.dot(m2, m3.T)
# print(result)
# print(m2[:, np.newaxis])
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.dot(M[0].T, M[1]))
print(M.shape)
print(M[:, :, np.newaxis].dot(M[:, np.newaxis, :]))
