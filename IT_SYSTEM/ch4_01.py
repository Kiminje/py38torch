import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time
from numba import cuda, jit

#   supervised learning
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
# print(x.shape, '\n', 0.8 * 569)
"""
(569, 30) 
 455.20000000000005
 -> it means the # of cancer data is 569 and each data has 30 parameters
 """
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

class LogisticNeuron:

    def __init__(self):
        #   z = Wx + b
        self.w = None
        self.b = None
        self.epsilon = None

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

    def fit(self, x, y, epochs, learningrate):
        start = time.time()
        self.w = np.ones(x.shape[1])  # 가중치를 초기화합니다.
        self.b = 0  # 절편을 초기화합니다.
        # self.epsilon = 0.1*np.sum(x.shape[1])/np.sum(x.shape)
        self.epsilon = learningrate / np.sum(x.shape[0])  # data 수만큼 scaling한다.
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
        return time.time() - start

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]  # 정방향 계산
        a = self.activation(np.array(z))  # 활성화 함수 적용
        return a > 0.5


# Epochs 에 따른 변화


@jit()
def collect_data(outfile):
    neuron = LogisticNeuron()
    EpochArr = [100, 500, 1000, 3000]
    TestRate = [0.7, 0.5, 0.3, 0.2, 0.1]
    LearnRate = [0.005, 0.01, 0.05, 0.2, 0.5]
    Index = [EpochArr, TestRate, LearnRate]
    np.array(Index)
    Result = []
    for i in EpochArr:
        perEpoch = []
        for rate in TestRate:
            perRate = []
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=rate, random_state=42)
            for lr in LearnRate:
                perlr = []
                perlr.append(lr)

                print("{} time!, {}test rate! {} learningrate".format(i, rate, lr))

                TempTime = neuron.fit(x_train, y_train, i, lr)
                perlr.append(TempTime)
                TempAcc = np.mean(neuron.predict(x_test) == y_test)
                perlr.append(TempAcc)
                print(TempAcc)
                perRate.append(perlr)

            np.savetxt(outfile, perRate, header="{} epoch   {} data rate, Check the Learning rate column 0".format(i, rate))
            perEpoch.append(perRate)
        Result.append(perEpoch)
        print(Result)
    return Result, Index


# EpochArr = [i * 100 for i in range(1, 101)]
with open('test.txt', 'w') as outfile:
    Result, Ind = collect_data(outfile)



#collect_data()
# plt.plot(EpArr, TimeArr)
# plt.show()
# plt.plot(EpArr, AccArr)
# plt.show()
#   print(np.mean(neuron.predict_ReLU(x_test) == y_test))
