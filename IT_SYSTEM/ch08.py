import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.signal import convolve
from scipy.signal import convolve2d
from scipy.signal import correlate
from scipy.signal import correlate2d
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#
# def relu(x):
#     return np.maximum(x,0)
#
# class ConvolutionNetwork:
#
#     def __init__(self, n_kernels=10, units=10, batch_size=32, learning_rate=0.1):
#         self.n_kernels = n_kernels  # 합성곱의 커널 개수
#         self.kernel_size = 3        # 커널 크기
#         self.optimizer = None       # 옵티마이저
#         self.conv_w = None          # 합성곱 층의 가중치
#         self.conv_b = None          # 합성곱 층의 절편
#         self.units = units          # 은닉층의 뉴런 개수
#         self.batch_size = batch_size  # 배치 크기
#         self.w1 = None              # 은닉층의 가중치
#         self.b1 = None              # 은닉층의 절편
#         self.w2 = None              # 출력층의 가중치
#         self.b2 = None              # 출력층의 절편
#         self.a1 = None              # 은닉층의 활성화 출력
#         self.losses = []            # 훈련 손실
#         self.val_losses = []        # 검증 손실
#         self.lr = learning_rate     # 학습률
#
#     def forpass(self, x):
#         # 3x3 합성곱 연산을 수행합니다.
#         c_out = tf.nn.conv2d(x, self.conv_w, strides=1, padding='SAME') + self.conv_b
#         # 렐루 활성화 함수를 적용합니다.
#         r_out = tf.nn.relu(c_out)
#         # 2x2 최대 풀링을 적용합니다.
#         p_out = tf.nn.max_pool2d(r_out, ksize=2, strides=2, padding='VALID')
#         # 첫 번째 배치 차원을 제외하고 출력을 일렬로 펼칩니다.
#         f_out = tf.reshape(p_out, [x.shape[0], -1])
#         z1 = tf.matmul(f_out, self.w1) + self.b1     # 첫 번째 층의 선형 식을 계산합니다
#         a1 = tf.nn.relu(z1)                          # 활성화 함수를 적용합니다
#         z2 = tf.matmul(a1, self.w2) + self.b2        # 두 번째 층의 선형 식을 계산합니다.
#         return z2
#
#     def init_weights(self, input_shape, n_classes):
#         g = tf.initializers.glorot_uniform()
#         self.conv_w = tf.Variable(g((3, 3, 1, self.n_kernels)))
#         self.conv_b = tf.Variable(np.zeros(self.n_kernels), dtype=float)
#         n_features = 14 * 14 * self.n_kernels
#         self.w1 = tf.Variable(g((n_features, self.units)))          # (특성 개수, 은닉층의 크기)
#         self.b1 = tf.Variable(np.zeros(self.units), dtype=float)    # 은닉층의 크기
#         self.w2 = tf.Variable(g((self.units, n_classes)))           # (은닉층의 크기, 클래스 개수)
#         self.b2 = tf.Variable(np.zeros(n_classes), dtype=float)     # 클래스 개수
#
#     def fit(self, x, y, epochs=100, x_val=None, y_val=None):
#         self.init_weights(x.shape, y.shape[1])    # 은닉층과 출력층의 가중치를 초기화합니다.
#         self.optimizer = tf.optimizers.SGD(learning_rate=self.lr)
#         # epochs만큼 반복합니다.
#         for i in range(epochs):
#             print('에포크', i, end=' ')
#             # 제너레이터 함수에서 반환한 미니배치를 순환합니다.
#             batch_losses = []
#             for x_batch, y_batch in self.gen_batch(x, y):
#                 print('.', end='')
#                 self.training(x_batch, y_batch)
#                 # 배치 손실을 기록합니다.
#                 batch_losses.append(self.get_loss(x_batch, y_batch))
#             print()
#             # 배치 손실 평균내어 훈련 손실 값으로 저장합니다.
#             self.losses.append(np.mean(batch_losses))
#             # 검증 세트에 대한 손실을 계산합니다.
#             self.val_losses.append(self.get_loss(x_val, y_val))
#
#     # 미니배치 제너레이터 함수
#     def gen_batch(self, x, y):
#         bins = len(x) // self.batch_size                   # 미니배치 횟수
#         indexes = np.random.permutation(np.arange(len(x))) # 인덱스를 섞습니다.
#         x = x[indexes]
#         y = y[indexes]
#         for i in range(bins):
#             start = self.batch_size * i
#             end = self.batch_size * (i + 1)
#             yield x[start:end], y[start:end]   # batch_size만큼 슬라이싱하여 반환합니다.
#
#     def training(self, x, y):
#         m = len(x)                    # 샘플 개수를 저장합니다.
#         with tf.GradientTape() as tape:
#             z = self.forpass(x)       # 정방향 계산을 수행합니다.
#             # 손실을 계산합니다.
#             loss = tf.nn.softmax_cross_entropy_with_logits(y, z)
#             loss = tf.reduce_mean(loss)
#
#         weights_list = [self.conv_w, self.conv_b,
#                         self.w1, self.b1, self.w2, self.b2]
#         # 가중치에 대한 그래디언트를 계산합니다.
#         grads = tape.gradient(loss, weights_list)
#         # 가중치를 업데이트합니다.
#         self.optimizer.apply_gradients(zip(grads, weights_list))
#
#     def predict(self, x):
#         z = self.forpass(x)                 # 정방향 계산을 수행합니다.
#         return np.argmax(z.numpy(), axis=1) # 가장 큰 값의 인덱스를 반환합니다.
#
#     def score(self, x, y):
#         # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환합니다.
#         return np.mean(self.predict(x) == np.argmax(y, axis=1))
#
#     def get_loss(self, x, y):
#         z = self.forpass(x)                 # 정방향 계산을 수행합니다.
#         # 손실을 계산하여 저장합니다.
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, z))
#         return loss.numpy()

(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.1)
# print(x_train[0])
y_test = tf.keras.utils.to_categorical(y_test)
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)
x_train = x_train.reshape(-1, 28, 28, 1)
# print(x_train[0])
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_test = x_test.astype('float32')
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /= 255
#
# cn = ConvolutionNetwork(n_kernels=10, units=100, batch_size=128, learning_rate=0.01)
# cn.fit(x_train, y_train_encoded, x_val=x_val, y_val=y_val_encoded, epochs=20)
#
# plt.plot(cn.losses)
# plt.plot(cn.val_losses)
# plt.ylabel('loss')
# plt.xlabel('iteration')
# plt.legend(['train_loss', 'val_loss'])
# plt.savefig('ch08_01.png')
# plt.clf()

# print(cn.score(x_val, y_val_encoded))
#
# conv1 = tf.keras.Sequential()
# conv1.add(Conv2D(10, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
# conv1.add(MaxPooling2D((2, 2)))
# conv1.add(Flatten())
# conv1.add(Dense(100, activation='relu'))
# conv1.add(Dense(10, activation='softmax'))
# conv1.summary()
#
# conv1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = conv1.fit(x_train, y_train_encoded, epochs=20, validation_data=(x_val, y_val_encoded))
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train_loss', 'val_loss'])
# plt.savefig('ch08_02.png')
# plt.clf()
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train_accuracy', 'val_accuracy'])
# plt.savefig('ch08_03.png')
# plt.clf()
#
# loss, accuracy = conv1.evaluate(x_val, y_val_encoded, verbose=0)
# print(accuracy)
ChNums = [64]
layers = [10]
Ratios = [0.25]
i = 1e-5
j = 1e-5
for Ch in ChNums:
    for Layer in layers:
        Split = Layer // 2
        for ratio in Ratios:
            conv2 = tf.keras.Sequential()
            conv2.add(Conv2D(Ch, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
            for _ in range(Split):
                conv2.add(Conv2D(Ch, (3, 3), activation='relu', padding='same'))
            conv2.add(MaxPooling2D((2, 2)))
            for _ in range(Split):
                conv2.add(Conv2D(Ch, (3, 3), activation='relu', padding='same'))
            conv2.add(MaxPooling2D((2, 2)))
            conv2.add(Flatten())
            conv2.add(Dropout(ratio))
            conv2.add(Dense(1024, activation='relu'))
            conv2.add(Dropout(ratio))
            conv2.add(Dense(512, activation='relu'))
            conv2.add(Dropout(ratio))
            conv2.add(Dense(64, activation='relu'))
            conv2.add(Dropout(ratio))
            conv2.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=i, l2=j)))

            conv2.summary()

            conv2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = conv2.fit(x_train, y_train_encoded, batch_size=32, epochs=30, validation_data=(x_val, y_val_encoded))
            print("training end!")
            loss, accuracy = conv2.evaluate(x_val, y_val_encoded, verbose=0)
            print(accuracy)

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train_loss', 'val_loss'])
            plt.grid(b=True, which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.title("<LOSS: {:0.3f}> Layer {}, Channel {}, Ratio: {}".format(loss, Layer, Ch, ratio))
            plt.savefig('ch08_loss{}_{}_{}.png'.format(Layer, Ch, ratio))
            plt.clf()

            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train_accuracy', 'val_accuracy'])
            plt.grid(b=True, which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.title("<Accuracy: {:0.3f} > Layer {}, Channel {}, Ratio: {}".format(accuracy, Layer, Ch, ratio))
            plt.savefig('ch08_acc{}_{}_{}.png'.format(Layer, Ch, ratio))
            plt.clf()

