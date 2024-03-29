import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y, test_size = 0.2) #, random_state = 42)

train_mean = np.mean(x_train, axis=0)
train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train-train_mean)/train_std
x_test_scaled = (x_test-train_mean)/train_std

class SingleLayer:
    def __init__(self):
        self.w = None
        self.b = None
        self.eta = None
        self.losses = []
        self.w_history = []
        self.b_history = []

    def forpass(self, x):
        z = np.dot(x , self.w) + self.b  # 직선 방정식을 계산합니다
        return z

    def backprop(self, x_k, err_k):
        w_grad_k = x_k * err_k    # 가중치에 대한 그래디언트를 계산합니다
        b_grad_k = 1 * err_k    # 절편에 대한 그래디언트를 계산합니다
        return w_grad_k, b_grad_k

    def activation(self, z_k):
        z_k = np.clip(z_k, -100, None) # 안전한 np.exp() 계산을 위해
        a_k = 1.0 / (1.0 + np.exp(-z_k))  # 시그모이드 계산
        return a_k
        
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])      # 가중치를 초기화합니다.
        self.b = 0.0                        # 절편을 초기화합니다.
        self.eta = 1.0
        self.w_history.append(self.w.copy())
        self.b_history.append(self.b)

        for i in range(epochs):           # epochs만큼 반복합니다
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))

            for k in indexes:    # 모든 샘플에 대해 반복합니다
                z_k = self.forpass(x[k])     # 정방향 계산
                a_k = self.activation(z_k)    # 활성화 함수 적용
                err_k = -(y[k] - a_k)          # 오차 계산
                w_grad_k, b_grad_k = self.backprop(x[k], err_k) # 역방향 계산
                self.w -= self.eta*w_grad_k
                self.b -= self.eta*b_grad_k
                a_k = np.clip(a_k, 1e-10, 1-1e-10)
                loss += -y[k]*np.log(a_k)-(1-y[k])*np.log(1-a_k)

            self.losses.append(loss/x.shape[0])
            self.w_history.append(self.w.copy())
            self.b_history.append(self.b)
    
    def predict(self, x):
        z = [self.forpass(x_k) for x_k in x]    # 정방향 계산
        a = self.activation(np.array(z))        # 활성화 함수 적용
        return a > 0.5
    
    def score(self, x, y):
        return np.mean(self.predict(x)==y)

single = SingleLayer()
single.fit(x_train, y_train)
print(single.score(x_test, y_test))

single_scaled = SingleLayer()
single_scaled.fit(x_train_scaled, y_train)
print(single_scaled.score(x_test_scaled, y_test))

plt.plot(single.losses)
plt.plot(single_scaled.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()