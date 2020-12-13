# import pandas as pd
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import add
from keras.layers import Add
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
import random
import pandas as pd
from pathlib import Path
from PIL import Image
from glob import glob
# from fastai.vision.all import *
# from fastai.metrics import error_rate
print(tf.keras.__version__)
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

import random
from PIL import ImageEnhance, ImageChops
def AugImage(image):
    Seed = random.random()
    Seed1 = Seed* 1.5 + 0.25
    enhancer = ImageEnhance.Brightness(image)
    brightness_image = enhancer.enhance(Seed1)

    #좌우 대칭
    if(Seed > 0.3):
        horizonal_flip_image = brightness_image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        horizonal_flip_image = brightness_image

    #상하 대칭
    if Seed < 0.7:
        vertical_flip_image = horizonal_flip_image.transpose(Image.FLIP_TOP_BOTTOM)
    else :
        vertical_flip_image = horizonal_flip_image

    #좌우 이동
    width, height = image.size
    shift = random.randint(0, width / 4) - 12
    horizonal_shift_image = ImageChops.offset(vertical_flip_image, shift, 0)
    horizonal_shift_image.paste((0), (0, 0, shift, height))

    #상하 이동
    width, height = image.size
    shift = random.randint(0, height /4) - 12
    vertical_shift_image = ImageChops.offset(horizonal_shift_image, 0, shift)
    vertical_shift_image.paste((0), (0, 0, width, shift))

    #회전
    Seed3 = Seed * 120
    rotate_image = vertical_shift_image.rotate(Seed3)

    #기울기
    #cx, cy = 0.1, 0
    #cx, cy = 0, 0.1
    Seed4 = Seed * 0.2
    cx, cy = 0, Seed4
    shear_image = rotate_image.transform(
        image.size,
        method=Image.AFFINE,
        data=[1, cx, 0,
              cy, 1, 0,])

    #확대 축소
    Seed5 = (Seed -0.5) + 1.1
    zoom = Seed5 #0.7 ~ 1.3
    width, height = image.size
    x = width / 2
    y = height / 2
    crop_image = shear_image.crop((x - (width / 2 / zoom), y - (height / 2 / zoom), x + (width / 2 / zoom), y + (height / 2 / zoom)))
    zoom_image = crop_image.resize((width, height), Image.LANCZOS)
    return zoom_image

"""
data: input to the residual module
K: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn K/4 filters)
stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
chanDim: defines the axis which will perform batch normalization
red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
reg: applies regularization strength for all CONV layers in the residual module
bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
bnMom: controls the momentum for the moving average
"""
# ImageDataLoaders
class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False,
                        reg=1e-4, bnEps=2e-5, bnMom=0.9):
        print(reg, stride, chanDim, data)
        shortcut = data
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act1)
        """
        Notice that the bias term is turned off for the CONV layer, 
        as the biases are already in the following BN layers 
        so there’s no need for a second bias term.        
        """
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False,
                              kernel_regularizer=l2(reg))(act1)
        d1 = Add()
        x = d1([conv3, shortcut])

        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        inputShape = (height, width, depth)
        chanDim = -1
        print(reg, stages, filters)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(inputs)

        x = Conv2D(filters[0], (5, 5), use_bias=False,
                   padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1,1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        for i in range(0, len(stages)):
            stride = (1, 1) if i ==0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                                       chanDim, red=True, bnEps=bnEps, bnMom=bnMom)
            print("x shape:", x)

            for j in range(0, stages[i] - 1):
                x = ResNet.residual_module(x, filters[i + 1],
                                           (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)


        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)
        # print(x)
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        print(x)
        x = Activation("softmax")(x)
        model = Model(inputs, x, name="resnet")
        return model


bs = 64 #   batch size
resize_size = 96 #  for training

import matplotlib.pyplot as plt
from PIL import Image
import Augmentor
import os

img_height = 96
img_width = 96
batch_size = 128
import pathlib

dir = "/home/inje/py38torch/Machine_Learning/kaggle_bee_vs_wasp"

if os.path.exists("./beevswasp_train.npz"):
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    C = np.load("./beevswasp_val.npz")
    X_val, y_val = C['x'], C['y']
    B = np.load("./beevswasp_train.npz")
    X_train, y_train = B['x'], B['y']
    A = np.load("./beevswasp_test.npz")
    X_test, y_test = A['x'], A['y']
    print(X_train.shape, y_train[0])
    X_train = X_train.astype(float) / 255
    X_val = X_val.astype(float) / 255
    X_test = X_test.astype(float) / 255
    # y_train = Augmentor.Pipeline.categorical_labels(y_train)
    # y_test = Augmentor.Pipeline.categorical_labels(y_test)
    # y_val = Augmentor.Pipeline.categorical_labels(y_val)
    # y_train = tf.keras.utils.to_categorical(y_train)
    # y_val = tf.keras.utils.to_categorical(y_val)
    # y_test = tf.keras.utils.to_categorical(y_test)
    print(X_train.shape, y_train[0])
    # (7939, 96, 96, 3)(7939, 4, 2)
    np.random.seed(42)  # for reproducibility
    # model = ResNet()
    # p = Augmentor.Pipeline()
    # p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    # p.flip_left_right(probability=0.5)
    # p.flip_top_bottom(probability=0.5)
    # p.rotate90(probability=0.5)
    # p.rotate270(probability=0.5)
    # p.crop_random(probability=1, percentage_area=0.5)
    # g = p.keras_generator_from_array(X_train, y_train, batch_size=100)
    # p = Augmentor.Pipeline()
    # p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    # p.flip_left_right(probability=0.5)
    # p.zoom_random(probability=0.5, percentage_area=0.8)
    # p.flip_top_bottom(probability=0.5)
    # p.crop_random(probability=1, percentage_area=0.5)
    # p.resize(probability=1.0, width=96, height=96)
    # g = p.keras_generator_from_array(X_train, y_train, batch_size=batch_size)
    model = ResNet.build(96, 96, 3, 4, (3, 4, 6), (64, 128, 256, 512), reg=0.0005)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    # history = model.fit_generator(g, steps_per_epoch=len(X_train) / batch_size,
    #                               epochs=50, validation_data=(X_val, y_val), verbose=1)
    history = model.fit(X_train, y_train, epochs=50,
                        validation_data=(X_val, y_val), batch_size=64, verbose=2)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(accuracy)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'])
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title("<LOSS> test: {:0.4f}".format(loss))
    plt.savefig('BeeVS_Wasp_loss_aug1.png')
    plt.clf()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'val_accuracy'])
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title("<ACC> test: {:0.4f}".format(accuracy))
    plt.savefig('BeeVS_Wasp_acc_aug1.png')
    plt.clf()


else:

    data_pat = Path('./kaggle_bee_vs_wasp')
    data_labels = pd.read_csv("/home/inje/py38torch/Machine_Learning/kaggle_bee_vs_wasp/labels.csv")
    # data_labels = data_labels.set_index('id')
    col = data_labels.id
    data_labels = data_labels.reindex(np.random.permutation(data_labels.index))
    # print(data_labels[1])
    # print(col[0:5])
    print(data_labels.is_bee[0: 10])
    train = data_labels[(data_labels.is_validation == 0) & (data_labels.is_final_validation == 0)]
    val = data_labels[data_labels.is_validation == 1]
    test = data_labels[data_labels.is_final_validation == 1]


    trainImg = []
    trainLabel = []
    valImg = []
    valLabel = []
    testImg = []
    testLabel = []
    arr0 = [1, 0, 0, 0]
    arr1 = [0, 1, 0, 0]
    arr2 = [0, 0, 1, 0]
    arr3 = [0, 0, 0, 1]
    for i, j in zip(train.path, train.label) :
        imgpath = dir + "/" + i
        # print(imgpath, j)
        img = Image.open(imgpath)
        img = img.convert("RGB")
        img = img.resize((img_height, img_height))

        data = np.asarray(img)
        trainImg.append(data)
        # trainLabel.append(j)
        if j == 'bee':
            for _ in range(0, 10):
                tmp = AugImage(img)
                data1 = np.asarray(tmp)
                trainImg.append(data1)
                trainLabel.append(arr0)
            trainLabel.append(arr0)
        elif j == 'wasp':
            for _ in range(0, 10):
                tmp = AugImage(img)
                data1 = np.asarray(tmp)
                trainImg.append(data1)
                trainLabel.append(arr1)
            trainLabel.append(arr1)
        elif j == 'insect':
            for _ in range(0, 10):
                tmp = AugImage(img)
                data1 = np.asarray(tmp)
                trainImg.append(data1)
                trainLabel.append(arr2)
            trainLabel.append(arr2)
        else:
            for _ in range(0, 10):
                tmp = AugImage(img)
                data1 = np.asarray(tmp)
                trainImg.append(data1)
                trainLabel.append(arr3)
            trainLabel.append(arr3)

    for i, j in zip(val.path, val.label):
        imgpath = dir + "/" + i
        img = Image.open(imgpath)
        img = img.convert("RGB")
        img = img.resize((img_height, img_height))
        data = np.asarray(img)
        valImg.append(data)
        if j == 'bee':
            for _ in range(0, 10):
                tmp = AugImage(img)
                data1 = np.asarray(tmp)
                valImg.append(data1)
                valLabel.append(arr0)
            valLabel.append(arr0)
        elif j == 'wasp':
            for _ in range(0, 10):
                tmp = AugImage(img)
                data1 = np.asarray(tmp)
                valImg.append(data1)
                valLabel.append(arr1)

            valLabel.append(arr1)
        elif j == 'insect':
            for _ in range(0, 10):
                tmp = AugImage(img)
                data1 = np.asarray(tmp)
                valImg.append(data1)
                valLabel.append(arr2)
            valLabel.append(arr2)
        else:
            for _ in range(0, 10):
                tmp = AugImage(img)
                data1 = np.asarray(tmp)
                valImg.append(data1)
                valLabel.append(arr3)
            valLabel.append(arr3)
        # valLabel.append(j)

    for i, j in zip(test.path, test.label):
        imgpath = dir + "/" + i
        img = Image.open(imgpath)
        img = img.convert("RGB")
        img = img.resize((img_height, img_height))
        data = np.asarray(img)
        testImg.append(data)
        if j == 'bee':
            testLabel.append(arr0)
        elif j == 'wasp':
            testLabel.append(arr1)
        elif j == 'insect':
            testLabel.append(arr2)
        else:
            testLabel.append(arr3)
        # testLabel.append(j)

    X1 = np.array(trainImg)

    y1 = np.array(trainLabel)
    print(y1[0:5])

    np.savez("./beevswasp_train.npz", x=X1, y=y1)
    print("done", len(y1), len(X1))
    X1 = np.array(valImg)
    y1 = np.array(valLabel)

    np.savez("./beevswasp_val.npz", x=X1, y=y1)
    print("done", len(y1), len(X1))

    X1 = np.array(testImg)
    y1 = np.array(testLabel)

    np.savez("./beevswasp_test.npz", x=X1, y=y1)
    print("done", len(y1))

