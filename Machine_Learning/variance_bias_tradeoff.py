import numpy as np
import random
from math import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
x = np.arange(0, 6 * pi, pi / 100)
y = np.sin(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state = 42)

PRINT = 2000


class Tradeoff():
    w1 = None
    b = 0.0
    len = None
    eta = 0.00001
    losses = []
    Variance = []
    Bias = []

    def backpropagation(self, x, err):
        w_grad = np.mean(np.multiply(-2 * err, x))
        b_grad = np.mean(-2 * err)
        return w_grad, b_grad

    def square(self, hat, y):
        expect = np.sum(hat) / self.len
        y_mean = np.sum(y) / self.len
        # Variance Term
        Var = np.sum(np.square(hat - expect)) / self.len
        # Bias Term
        Bias = np.sum(np.square(hat - y_mean)) / self.len
        return Var, Bias

    def fit_linear(self, x, y, epochs=100000):
        print("*************linear*************")
        self.len = len(x)
        self.w1 = np.zeros(x[0].shape)
        for i in range(epochs):
            b_grad = 0.0
            w_grad = np.zeros(x[0].shape)

            y_hat = np.dot(x, self.w1) + self.b
            err = y - y_hat

            loss = np.sum(np.square(err))
            w_grad, b_grad = self.backpropagation(x, err)

            self.w1 -= w_grad * self.eta
            self.b -= b_grad * self.eta
            Var, Bias = self.square(y_hat, y)

            self.losses.append(loss)
            self.Bias.append(Bias)
            self.Variance.append(Var)

            if i % PRINT == 0:
                y_hat = np.dot(x, self.w1) + self.b

                print("iter %d" % i, end='\t')
                print("loss : %3f"% loss, end='\t')

                print("variance = %3f" % Var, end='\t')
                print("bias = %3f" % Bias)

    def fit_constant(self, x, y, epochs=100000):
        print("*************constant*************")
        self.len = len(x)
        # self.w1 = np.zeros(x[0].shape)
        for i in range(epochs):
            b_grad = 0.0
            # w_grad = np.zeros(x[0].shape)

            y_hat = self.b
            err = y - y_hat
            loss = np.sum(np.square(err))
            b_grad = -2 * err
            # print(np.square(err))

            # self.w1 -= w_grad * self.eta
            self.b -= b_grad * self.eta

            if i % PRINT == 0:
                y_hat = self.b

                print("iter %d" % i, end='\t')
                print("loss : %3f"% loss, end='\t')
                Var, Bias = self.square(y_hat, y)
                print("variance = %3f" % Var, end='\t')
                print("bias = %3f" % Bias)

    def Score(self, x, y, Linear=True):
        if Linear:
            y_hat = x * self.w1 + self.b
            Var, Bias = self.square(y_hat, y)
            print("variance = %3f" % Var, end='\t')
            print("bias = %3f" % Bias)
        else:
            y_hat = self.b
            Var, Bias = self.square(y_hat, y)
            print("variance = %3f" % Var, end='\t')
            print("bias = %3f" % Bias)
    # def plot(self):

    def plot(self):
        # plt.plot(self.losses, color='red')
        plt.plot(self.Variance, color='green')

        plt.plot(self.Bias, color='blue')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()



Neuron = Tradeoff()
Neuron.fit_linear(x_train, y_train)
Neuron.Score(x_test, y_test)
Neuron.plot()

Neuron_cons = Tradeoff()
Neuron_cons.fit_constant(x_train, y_train)
Neuron_cons.Score(x_test, y_test, Linear=False)
Neuron_cons.plot()
