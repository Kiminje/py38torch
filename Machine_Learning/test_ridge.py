import numpy as np

np.random.seed(123)
dim = 5
x = 2 * np.random.rand(100,dim)
coef = 10 * np.random.rand(1,dim)
y = -8 + np.dot(x, coef.T)+np.random.randn(100,1)

print( x[1].shape, y.shape, coef)
w1 = np.random.rand(dim, 1)
w0 = np.random.rand(1, 1)
Epochs = 2001
N = len(y[:])
print(N, w1.shape)
learning_rate = 0.05 / N  # N개의 데이터에 대해 학습하기 때문에 1회 반복에 고정된 Learning Rate 사용
Alpha = 1.0
for ind in range(Epochs):

    for i in range(N):
        y_pred = np.dot(x[i], w1) + w0
        diff = y[i] - y_pred
        # w0_update를 dot 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 생성
        w0_factors = np.ones((1, 1))
        # w1과 w0을 업데이트할 w1_update와 w0_update 계산
        w1_update = -2 * learning_rate * diff[0] * x[i] + 2 * Alpha * w1.T * learning_rate

        w0_update = -2 * learning_rate * w0_factors * diff

    w1 = w1 - w1_update.T
    w0 = w0 - w0_update

    y_pred = np.dot(x, w1) + w0

    # print("w1", w1.shape, "w0: ", w0.shape, "w1_updateshpae", w1_update.shape)
    cost = np.sum(np.square(y - y_pred)) / N + (Alpha * np.sum(np.dot(w1.T, w1))) / N
    print('Gradient Descent Total Cost:{0:.4f}'.format(cost))
    # break
print("real coef is ", coef)
print("trained weight : ", w1)
print("trained intercept: ", w0)
y_mean = np.sum(y) / N
R_square = np.sum(np.square(y_pred - y_mean)) / np.sum(np.square(y - y_mean))
print("R_square score : ", R_square)
rmse = np.sqrt(np.sum(np.square(y- y_pred)) / N)
print("RMSE : ", rmse)




