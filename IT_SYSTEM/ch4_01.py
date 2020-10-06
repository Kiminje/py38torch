import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y, test_size = 0.2) #, random_state = 42)

class LogisticNeuron:
    
    def __init__(self):
        self.w = None
        self.b = None
        self.epsilon = None

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b  # 직선 방정식을 계산합니다
        return z

    def backprop(self, x, err):
        w_grad = x * err    # 가중치에 대한 그래디언트를 계산합니다
        b_grad = 1 * err    # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None) # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a
        
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])      # 가중치를 초기화합니다.
        self.b = 0                        # 절편을 초기화합니다.
        self.epsilon = 0.1*np.sum(x.shape[1])/np.sum(x.shape)

        for i in range(epochs):           # epochs만큼 반복합니다
            w_grad = np.zeros(x.shape[1])
            b_grad = 0.0

            for x_i, y_i in zip(x, y):    # 모든 샘플에 대해 반복합니다
                z = self.forpass(x_i)     # 정방향 계산
                a = self.activation(z)    # 활성화 함수 적용
                err = -(y_i - a)          # 오차 계산
                w_grad_i, b_grad_i = self.backprop(x_i, err) # 역방향 계산
                w_grad += w_grad_i
                b_grad += b_grad_i

            self.w -= self.epsilon*w_grad          # 가중치 업데이트
            self.b -= self.epsilon*b_grad          # 절편 업데이트
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]    # 정방향 계산
        a = self.activation(np.array(z))        # 활성화 함수 적용
        return a > 0.5

neuron = LogisticNeuron()
neuron.fit(x_train, y_train)
print(np.mean(neuron.predict(x_test) == y_test))