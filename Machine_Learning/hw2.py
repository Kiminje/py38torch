import numpy as np
import numpy.linalg as lin
#1. similar matrix vs matrix
A = np.array([[-5, -6, 6], [-9, -8, 12], [-12, -12, 16]])
A_hat = np.array([[-5, -15/2, -3], [-15/2, -8, 0], [3, 0, 16]])

print(lin.eigvals(A), "\n", lin.eigvals(A_hat))
