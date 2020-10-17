
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(123)
x = 2 * np.random.rand(100,1)
y = -8 +2 * x+np.random.randn(100,1)

#그래프로 나타내 봅니다.
plt.figure(figsize=(8,5))
plt.scatter(x, y)
plt.show()

w0 = np.zeros((1, 1))
w1 = np.zeros((1, 1))
iters = 2001
N = len(y)
learning_rate = 0.05

for ind in range(iters):
    y_pred = np.dot(x, w1.T) + w0
    diff = y - y_pred

    # w0_update를 dot 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 생성
    w0_factors = np.ones((N, 1))

    # w1과 w0을 업데이트할 w1_update와 w0_update 계산
    w1_update = -(2 / N) * learning_rate * (np.dot(x.T, diff))
    w0_update = -(2 / N) * learning_rate * (np.dot(w0_factors.T, diff))

    w1 = w1 - w1_update
    w0 = w0 - w0_update
    y_pred = w1[0, 0] * x + w0

    print("w1:{0:.3f} w0:{1:.3f}".format(w1[0, 0], w0[0, 0]))
    cost = np.sum(np.square(y - y_pred)) / N
    print('Gradient Descent Total Cost:{0:.4f}'.format(cost))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fit the linear regression model
linreg = LinearRegression()
linreg.fit(x, y)


y_pred1 = linreg.predict(x)


# Evaluate the model
rmse_train = np.sqrt(mean_squared_error(y, y_pred1))

r2_train = linreg.score(x, y_pred1)

print("Train RMSE: ", rmse_train)

print("Train R^2: ", r2_train)
beta1=linreg.coef_
beta0=linreg.intercept_
print('beta1',beta1,"beta0",beta0)

#정규방정식 이용
X = np.c_[np.ones(len(x)), x[:,0]] #    column, column
print("X is ", X)
print(x)
print("x[:,0]", x[:, 0]) # transpose
beta_v3 = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
print(beta_v3)
y_pred = X@beta_v3
rmse_train = np.sqrt(mean_squared_error(y, y_pred1))
print(rmse_train)

#과제
from sklearn.linear_model import Ridge, Lasso




alpha_value=0.1
model = Ridge(alpha = alpha_value)
"""Fit Ridge regression model.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        Returns
        -------
        self : returns an instance of self.
        """

model.fit(x,y)
y_pred2= model.predict(x)
rmse_ridge= np.sqrt(mean_squared_error(y, y_pred2))
print("model spec:", model.coef_, model.intercept_)
print("rmse ridge", rmse_ridge)

import time

class LogisticNeuron:

    def __init__(self):
        #   z = Wx + b
        self.w = None
        self.b = None
        self.epsilon = None
        self.alpha = None

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b  # 직선 방정식을 계산합니다
        return z

    def backprop(self, x, err):
        w_grad = x * err  # 가중치에 대한 그래디언트를 계산합니다 dz/dW = x, x * err
        b_grad = 1 * err  # 절편에 대한 그래디언트를 계산합니다 dz/db = 1, 1 * err
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)  # 안전한 np.exp() 계산을 위해
        #   numpy.clip(a, a_min, a_max, out=None, **kwargs), z >= -100
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a

    def activate_ReLU(self, z):
        a = np.maximum(z, 0)
        return a

    def fit(self, x, y, epochs):
        start = time.time()
        self.w = np.ones(x.shape)  # 가중치를 초기화합니다.
        self.b = 0  # 절편을 초기화합니다.
        # self.epsilon = 0.1*np.sum(x.shape[1])/np.sum(x.shape)
        self.epsilon = 0.1   # data 수만큼 scaling한다.
        self.alpha = 1.0
        # print("this is the code for checking epsilon",np.sum(x.shape), np.sum(x.shape[1]))
        # 'epsilon' means learning rate and its value is (sum of parameters)
        # Learning rate is hyper parameter

        for i in range(epochs):  # epochs만큼 반복합니다
            w_grad = np.zeros(x.shape)
            # print(np.shape(w_grad))
            b_grad = 0.0
            # y_hat = np.dot(self.w, x) + self.b
            # obj_function = np.square(y_hat - y)

            for x_i, y_i in zip(x, y):  # 모든 샘플에 대해 반복합니다
                pred = np.dot(x_i, self.w.T) + self.b  # 정방향 계산

                # a = self.activation(z)  # 활성화 함수 적용
                err = np.dot((pred - y_i), (pred - y_i).T) + self.alpha * np.dot(self.w, self.w.T) # 오차 계산
                # print(err)
                w_grad_i, b_grad_i = self.backprop(x_i, err)  # 역방향 계산 derr/dx
                w_grad += w_grad_i
                b_grad += b_grad_i

            self.w -= self.epsilon * w_grad  # 가중치 업데이트
            self.b -= self.epsilon * b_grad  # 절편 업데이트
        return time.time() - start


    def fit_dynamic(self, x, y, epochs):
        start = time.time()
        self.w = np.ones(x.shape[1])  # 가중치를 초기화합니다.
        self.b = 0  # 절편을 초기화합니다.
        # self.epsilon = 0.1*np.sum(x.shape[1])/np.sum(x.shape)
        self.epsilon = LR / np.sum(x.shape[0])  # data 수만큼 scaling한다.
        # print("this is the code for checking epsilon",np.sum(x.shape), np.sum(x.shape[1]))
        # 'epsilon' means learning rate and its value is (sum of parameters)
        # Learning rate is hyper parameter

        for i in range(epochs):  # epochs만큼 반복합니다
            w_grad = np.zeros(x.shape[1])
            # print(np.shape(w_grad))
            b_grad = 0.0

            for x_i, y_i in zip(x, y):  # 모든 샘플에 대해 반복합니다
                z = self.forpass(x_i)  # 정방향 계산
                a = self.activation(z)  # 활성화 함수 적용
                err = -(y_i - a)  # 오차 계산
                # print(err)
                w_grad_i, b_grad_i = self.backprop(x_i, err)  # 역방향 계산 derr/dx
                w_grad += w_grad_i
                b_grad += b_grad_i

            self.w -= self.epsilon * w_grad  # 가중치 업데이트
            self.b -= self.epsilon * b_grad  # 절편 업데이트
            self.epsilon *= DYNAMIC
        return time.time() - start


    def fit_LR(self, x, y, epochs):
        start = time.time()
        self.w = np.ones(x.shape[1])  # 가중치를 초기화합니다.
        self.b = 0  # 절편을 초기화합니다.
        # self.epsilon = 0.1*np.sum(x.shape[1])/np.sum(x.shape)
        rate = []

        self.epsilon = LR / np.sum(x.shape[0])  # data 수만큼 scaling한다.
        # print("this is the code for checking epsilon",np.sum(x.shape), np.sum(x.shape[1]))
        # 'epsilon' means learning rate and its value is (sum of parameters)
        # Learning rate is hyper parameter

        for i in range(epochs):  # epochs만큼 반복합니다
            w_grad = np.zeros(x.shape[1])
            # print(np.shape(w_grad))
            b_grad = 0.0

            for x_i, y_i in zip(x, y):  # 모든 샘플에 대해 반복합니다
                z = self.forpass(x_i)  # 정방향 계산
                a = self.activation(z)  # 활성화 함수 적용
                err = -(y_i - a)  # 오차 계산
                # print(err)
                w_grad_i, b_grad_i = self.backprop(x_i, err)  # 역방향 계산 derr/dx
                w_grad += w_grad_i
                b_grad += b_grad_i

            self.w -= self.epsilon * w_grad  # 가중치 업데이트
            self.b -= self.epsilon * b_grad  # 절편 업데이트
            self.epsilon *= DYNAMIC
        end = time.time() - start


        return

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]  # 정방향 계산
        a = self.activation(np.array(z))  # 활성화 함수 적용
        return a > 0.5