import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

x = cancer.data # data에 다양한 feature가 들어있다.(성별, 키, 등등..)
y = cancer.target # 암에 걸렸냐 안걸렸냐

#   stratify 는 dataset의 비율을 유지하냐 안하냐
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size= 0.2, random_state=40)
# random state를 지정해줄 경우, n을 뽑는 위치가 고정된다. 즉 여러번 돌리던간에 결과값이 같다.
# 이를 이용해 다른 parameter를 변경시켰을 때, data를 뽑는 random에 따라 변한것이 아님을 체크할 수 있다.

print(cancer.feature_names, cancer.target_names)
print("x[1]= ", x[1], len(x[1]),"\n", "y[1]", y[1]) # 한사람의 data가 출력되고 유방암에 걸렸나 안걸렸나 확인
"""
plt.boxplot(cancer.data) # box plot -> 중간값을 중심으로, 중간값의 +-1/4로 box를 형성
#   # 중간값과 하단 1/4지점의 1.5배 지접으로 선으로 연결한다. 벗어나면 동그라미로 표시한다.
#   boxplot을 통해 편차가 큰 데이터를 쉽게 알 수 있다.
plt.show()
"""

class LogisticeNeuron:
    def __init__(self):
        # 변수만 선언하겠다. 나중에 data를 넣겠다. w의 dim이 얼마나 되는지 모른다.
        # 나중에 넣고 싶을 때, none을 쓰자.
        self.w = None
        self.b = None
        self.eta = None
    def forpass(self, x):
        z_k = np.sum(x*self.w) + self.b
        return z_k

    # err = 실제값에서 estimation을 뺀 값
    #실제로는 k 번째에 대한 gradient를 반환한다.
    def backprop(self, x, err_k):
        w_grad_k = err_k * x
        b_grad_k = err_k * 1
        return w_grad_k, b_grad_k

    # z값을 받아와 a를 return 해준다.
    def activation(self, z):
        # z가 -100보다 작아지지 않게 해라. 실제로 z가 너무 작으면 a에 대해 precision 문제가 발생할 수 있다.
        z = np.clip(z, -100, None)
        a = 1/(1 +np.exp(-z))
        return a

    def fit(self, x, y, epochs=100):
        # initialize

        # shape[0]은 data의 수, [1]은 column, feature 의 수를 넣어주고 싶으니 shape[1]
        self.w = np.ones(x.shape[1])
        self.b = 0.0
        # x.shape[0]으로 나눈 이유? 사람이 100명일때랑 10000명일 때랑 gradient계산할 때, eta가 100배는 차이가 나야지
        # learning rate 이 같게 된다.
        self.eta = 0.1 /(x.shape[0])
        # 인간이 직접 넣어줘야하는 부분 : hyper parameter

        for i in range(epochs):
            w_grad = np.zeros(x.shape[1])
            b_grad = 0.0
            
            #x_k = x[0], y_k = y[0]
            for x_k, y_k in zip(x, y):
                z_k = self.forpass(x_k)
                a_k = self.activation(z_k)
                err_k = -(y_k - a_k)
                w_grad_k, b_grad_k = self.backprop(x_k, err_k)
                w_grad = w_grad + w_grad_k
                b_grad += b_grad_k
            self.w = self.w - self.eta*w_grad
            self.b -= self.eta * b_grad
            
    def predict(self, x):
        z = [self.forpass(x_k) for x_k in x]
        # python array -> np.array
        a = self.activation(np.array(z))
        return a > 0.5
    
neuron = LogisticeNeuron()
neuron.fit(x_train, y_train)
print(np.mean(neuron.predict(x_test) == y_test))
arrtest = np.arange(100, 1500, 100)
print(arrtest)
# mean? 옳게 맞출 확률