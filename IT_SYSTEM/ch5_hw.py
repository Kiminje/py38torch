import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd


from enum import Enum as En

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)  # , random_state = 42)

train_mean = np.mean(x_train, axis=0)
train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - train_mean) / train_std
x_test_scaled = (x_test - train_mean) / train_std

PRINT_EVERY = 5000
PRINT_ANNEAL_EVERY = 30000
ANNEAL = False
Epochs = 20000

BatchSize = 10


# print(mode.BATCH)
# MODE = mode.BATCH
# print(MODE == 1)


class LogisticNeuron:

    def __init__(self):
        self.w = None
        self.b = None
        self.eta = None
        self.losses = []
        self.Scores = []
        self.w_history = []
        self.b_history = []
        self.Datalen = None

    def forpass(self, x):
        z = np.dot(x, self.w) + self.b  # x
        return z

    # aL / aW = aL/aY_hat * aY_hat/aZ * aZ/aW
    def backprop(self, x, y, a):
        Ypartial = -(y / a)
        Zpartial = np.multiply(a, (1 - a))
        Wpartial = x
        # print("shape info", Ypartial.shape, Wpartial.shape, Zpartial.shape)
        seq1 = np.dot(Wpartial.T, Zpartial * Ypartial)
        # print(seq1.shape)
        w_grad = seq1 / len(y)  # (dividing by num of data, )
        b_grad = np.sum(Ypartial * Zpartial) / len(y)
        # print(w_grad.shape, b_grad.shape)
        # w_grad_k = x_k * err_k  # 가중치에 대한 그래디언트를 계산합니다
        # b_grad_k = 1 * err_k  # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad

    def activation(self, z_k):
        z_k = np.clip(z_k, -100, None)  # 안전한 np.exp() 계산을 위해
        a_k = 1.0 / (1.0 + np.exp(-z_k))  # 시그모이드 계산
        return a_k

    def fit(self, x, y, epochs=Epochs, Shuffle=False, Batch=False, Size=BatchSize):
        self.w = np.ones(x.shape[1])  # 가중치를 초기화합니다.
        self.b = np.ones(1)  # 절편을 초기화합니다.
        self.eta = 0.0001
        self.w_history.append(self.w.copy())
        self.b_history.append(self.b)
        self.Datalen = len(y)
        # print(self.Datalen)

        for i in range(epochs):  # epochs만큼 반복합니다

            if Shuffle == True:
                x, y = shuffle(x, y)
                if Batch:
                    random_indices = np.random.choice(self.Datalen, size=Size, replace=False)
                    y_s = y[random_indices]
                    x_s = x[random_indices]
                else:
                    x_s, y_s = x, y
            else:
                x_s, y_s = x, y

            w_grad = np.zeros(x_s.shape[1])
            # print("wgradshape", w_grad.shape)
            # print("x_s shape", x_s.shape)

            b_grad = 0.0
            loss = 0
            z = self.forpass(x_s)  # 정방향 계산
            a = self.activation(z)  # sigmoid
            sub = y_s - a  # y - y_hat
            loss = np.sum(-np.multiply(y_s, np.log(a))) / len(y_s)  # cross entropy

            w_grad, b_grad = self.backprop(x_s, y_s, a)

            self.w -= self.eta * w_grad  # 가중치 업데이트
            self.b -= self.eta * b_grad  # 절편 업데이트

            partial_score = self.score(x, y)
            self.Scores.append(partial_score)
            if i % PRINT_EVERY == 0:
                print("shape is ", x_s.shape, y_s.shape)
                if i % PRINT_ANNEAL_EVERY == 0:
                    self.eta *= 0.5
                    print("learning rate %4f" % self.eta)
                print("iter : %d" % i, end='\t')
                print("loss %5f" % loss, end='\t')
                print("score: %4f" % partial_score)

            # if i > 300:
            self.losses.append(loss)

            self.w_history.append(self.w.copy())
            self.b_history.append(self.b)

    def predict(self, x):
        z = self.forpass(x)  # 정방향 계산
        a = self.activation(np.array(z))  # 활성화 함수 적용
        return a > 0.5

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


# neuron = LogisticNeuron()
# neuron.fit(x_train, y_train)
# print(neuron.score(x_test, y_test))
neuron_scaled = LogisticNeuron()
neuron_scaled.fit(x_train_scaled, y_train)
print(neuron_scaled.score(x_test_scaled, y_test))

neuron_scaled_shuffle = LogisticNeuron()
neuron_scaled_shuffle.fit(x_train_scaled, y_train, Shuffle=True, Batch=True)
print(neuron_scaled_shuffle.score(x_test_scaled, y_test))
# plt.plot(neuron.losses, color='green')
# plt.plot(neuron_scaled.losses)
plt.plot(neuron_scaled.Scores, color='green')
# plt.plot(neuron_scaled_shuffle.losses, color='red')
plt.plot(neuron_scaled_shuffle.Scores, color='blue')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
