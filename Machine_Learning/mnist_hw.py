from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
X,y = mnist['data'], mnist['target']
print(X.shape) #    (70000, 784)
print(y.shape) #    (70000, )
import matplotlib as mlt
import matplotlib.pyplot as plt
# some_digit=X[0]
# some_digit_image=some_digit.reshape(28,28)
# plt.imshow(some_digit_image, cmap="binary")
# plt.show()
y[0]#문자형이므로 숫자형으로 바꾸어야 함
y=y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]