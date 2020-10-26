import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from time import time
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

train_mean = np.mean(x_train, axis=0)
train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - train_mean) / train_std
x_test_scaled = (x_test - train_mean) / train_std

PRINT_EVERY = 5000
PRINT_ANNEAL_EVERY = 30000
ANNEAL = False
Epochs = 20000

class LogisticNeuron:

    def __init__(self):
        self.w = None
        self.b = None
        self.eta = None
        self.Score = []
        self.losses = []
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
        seq1 = np.dot(Wpartial.T, Zpartial * Ypartial)
        # print(seq1.shape)
        w_grad = seq1 / self.Datalen   # (num of data, )
        b_grad = np.sum(Ypartial * Zpartial) / self.Datalen
        # w_grad_k = x_k * err_k  # 가중치에 대한 그래디언트를 계산합니다
        # b_grad_k = 1 * err_k  # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad

    def activation(self, z_k):
        z_k = np.clip(z_k, -100, None)  # 안전한 np.exp() 계산을 위해
        a_k = 1.0 / (1.0 + np.exp(-z_k))  # 시그모이드 계산
        return a_k

    def fit(self, x, y, epochs=Epochs, Shuffle=True):
        start = time()
        self.w = np.ones(x.shape[1])  # 가중치를 초기화합니다.
        self.b = np.ones(1)  # 절편을 초기화합니다.
        self.eta = 0.001
        self.w_history.append(self.w.copy())
        self.b_history.append(self.b)
        self.Datalen = x.shape[0]
        print(self.Datalen)

        for i in range(epochs):  # epochs만큼 반복합니다
            w_grad = np.zeros(x.shape[1])
            b_grad = 0.0
            loss = 0
            if Shuffle == True:
                x, y = shuffle(x, y)

            z = self.forpass(x)  # 정방향 계산
            a = self.activation(z)  # sigmoid
            sub = y - a  # y - y_hat
            loss = np.sum(-np.multiply(y, np.log(a))) / self.Datalen  # cross entropy
            # print(self.w, self.b)
            w_grad, b_grad = self.backprop(x, y, a)

            self.w -= self.eta * w_grad  # 가중치 업데이트
            self.b -= self.eta * b_grad  # 절편 업데이트
            partial = self.score(x, y)
            if i % PRINT_EVERY == 0:
                if i % PRINT_ANNEAL_EVERY == 0 and i != 0 and ANNEAL:
                    self.eta *= 0.5
                    print("learning rate %4f" % self.eta)
                print("iter : %d" %i, end='\t')
                print("loss %5f" % loss, end='\t')
                print("score: %4f" % partial)
            self.losses.append(loss)
            self.w_history.append(self.w.copy())
            self.b_history.append(self.b)
            self.Score.append(partial)
        end = time() - start
        print("executed time %3f" % end)

    def batch_fit(self, x, y, epochs=Epochs, Shuffle=True, Size=30):
        start = time()
        self.w = np.ones(x.shape[1])  # 가중치를 초기화합니다.
        self.b = np.ones(1)  # 절편을 초기화합니다.
        self.eta = 0.001
        self.w_history.append(self.w.copy())
        self.b_history.append(self.b)
        self.Datalen = Size

        for i in range(epochs):  # epochs만큼 반복합니다
            w_grad = np.zeros(x.shape[1])
            # print(w_grad.shape, self.w.shape)
            b_grad = 0.0
            loss = 0
            if Shuffle:
                x, y = shuffle(x, y)

            # random한 data를 batch size만큼 뽑아서 학습시키도록 한다.
            random_indices = np.random.choice(455, size=Size, replace=False)
            y_s = y[random_indices]
            x_s = x[random_indices]

            z = self.forpass(x_s)  # 정방향 계산
            a = self.activation(z)  # sigmoid
            sub = y_s - a  # y - y_hat
            loss = np.sum(-np.multiply(y_s, np.log(a))) / self.Datalen  # cross entropy
            # print(self.w, self.b)
            w_grad, b_grad = self.backprop(x_s, y_s, a)

            self.w -= self.eta * w_grad  # 가중치 업데이트
            self.b -= self.eta * b_grad  # 절편 업데이트
            partial = self.score(x, y)
            if i % PRINT_EVERY == 0:
                if (i % PRINT_ANNEAL_EVERY == 0) and i !=0 and ANNEAL:
                    self.eta *= 0.5
                    print("learning rate %4f" % self.eta)
                print("iter : %d" %i, end='\t')
                print("loss %5f" % loss, end='\t')
                print("score: %4f" % partial)
            self.losses.append(loss)
            self.w_history.append(self.w.copy())
            self.b_history.append(self.b)
            self.Score.append(partial)
        end = time() - start
        print("executed time %3f" % end)

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
neuron_scaled.fit(x_train_scaled, y_train, Shuffle=False)
print(neuron_scaled.score(x_test_scaled, y_test))

# plt.plot(neuron.losses, color='green')
# plt.plot(neuron_scaled.losses)

neuron_scaled_batch = LogisticNeuron()
neuron_scaled_batch.fit(x_train_scaled, y_train)
print(neuron_scaled_batch.score(x_test_scaled, y_test))
#
# neuron_scaled_batch1 = LogisticNeuron()
# neuron_scaled_batch1.batch_fit(x_train_scaled, y_train, Size=455)
# print(neuron_scaled_batch1.score(x_test_scaled, y_test))

# plt.plot(neuron.losses, color='green')
plt.plot(neuron_scaled.losses, color='red')
plt.plot(neuron_scaled_batch.losses, color='green')
# plt.plot(neuron_scaled_batch1.losses, color='blue')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()