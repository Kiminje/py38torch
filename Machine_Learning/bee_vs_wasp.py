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
import pickle
import joblib

import xgboost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, \
    mean_squared_error, classification_report
# print(tf.keras.__version__)
# gpus = tf.config.experimental.list_physical_devices('GPU')
from sklearn.svm import SVC

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
    Seed1 = Seed * 1.5 + 0.25
    enhancer = ImageEnhance.Brightness(image)
    brightness_image = enhancer.enhance(Seed1)

    # 좌우 대칭
    if (Seed > 0.3):
        horizonal_flip_image = brightness_image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        horizonal_flip_image = brightness_image

    # 상하 대칭
    if Seed < 0.7:
        vertical_flip_image = horizonal_flip_image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        vertical_flip_image = horizonal_flip_image

    # 좌우 이동
    width, height = image.size
    shift = random.randint(0, width / 4) - 12
    horizonal_shift_image = ImageChops.offset(vertical_flip_image, shift, 0)
    horizonal_shift_image.paste((0), (0, 0, shift, height))

    # 상하 이동
    width, height = image.size
    shift = random.randint(0, height / 4) - 12
    vertical_shift_image = ImageChops.offset(horizonal_shift_image, 0, shift)
    vertical_shift_image.paste((0), (0, 0, width, shift))

    # 회전
    Seed3 = Seed * 120
    rotate_image = vertical_shift_image.rotate(Seed3)

    # 기울기
    # cx, cy = 0.1, 0
    # cx, cy = 0, 0.1
    Seed4 = Seed * 0.2
    cx, cy = 0, Seed4
    shear_image = rotate_image.transform(
        image.size,
        method=Image.AFFINE,
        data=[1, cx, 0,
              cy, 1, 0, ])

    # 확대 축소
    Seed5 = (Seed - 0.5) + 1.1
    zoom = Seed5  # 0.7 ~ 1.3
    width, height = image.size
    x = width / 2
    y = height / 2
    crop_image = shear_image.crop(
        (x - (width / 2 / zoom), y - (height / 2 / zoom), x + (width / 2 / zoom), y + (height / 2 / zoom)))
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
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
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

#
# class MyModel:
#     @staticmethod
#     def residual_module(data, K, stride, chanDim, red=False,
#                         reg=1e-4, bnEps=2e-5, bnMom=0.9):
#         print(reg, stride, chanDim, data)
#         shortcut = data
#         bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
#         act1 = Activation("relu")(bn1)
#         conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
#                        kernel_regularizer=l2(reg))(act1)
#         """
#         Notice that the bias term is turned off for the CONV layer,
#         as the biases are already in the following BN layers
#         so there’s no need for a second bias term.
#         """
#         bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
#         act2 = Activation("relu")(bn2)
#         conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
#                        kernel_regularizer=l2(reg))(act2)
#
#         bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
#         act3 = Activation("relu")(bn3)
#         conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)
#
#         if red:
#             shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False,
#                               kernel_regularizer=l2(reg))(act1)
#         d1 = Add()
#         x = d1([conv3, shortcut])
#
#         return x
#
#     @staticmethod
#     def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):
#         inputShape = (height, width, depth)
#         chanDim = -1
#         print(reg, stages, filters)
#
#         if K.image_data_format() == "channels_first":
#             inputShape = (depth, height, width)
#             chanDim = 1
#
#         inputs = Input(shape=inputShape)
#         x = BatchNormalization(axis=chanDim, epsilon=bnEps,
#                                momentum=bnMom)(inputs)
#
#         x = Conv2D(filters[0], (5, 5), use_bias=False,
#                    padding="same", kernel_regularizer=l2(reg))(x)
#         x = BatchNormalization(axis=chanDim, epsilon=bnEps,
#                                momentum=bnMom)(x)
#         x = Activation("relu")(x)
#         x = ZeroPadding2D((1, 1))(x)
#         x = MaxPooling2D((3, 3), strides=(2, 2))(x)
#
#         for i in range(0, len(stages)):
#             stride = (1, 1) if i == 0 else (2, 2)
#             x = ResNet.residual_module(x, filters[i + 1], stride,
#                                        chanDim, red=True, bnEps=bnEps, bnMom=bnMom)
#             print("x shape:", x)
#
#             for j in range(0, stages[i] - 1):
#                 x = ResNet.residual_module(x, filters[i + 1],
#                                            (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)
#
#         x = BatchNormalization(axis=chanDim, epsilon=bnEps,
#                                momentum=bnMom)(x)
#         x = Activation("relu")(x)
#         x = AveragePooling2D((8, 8))(x)
#         # print(x)
#         x = Flatten()(x)
#         x = Dense(classes, kernel_regularizer=l2(reg))(x)
#         print(x)
#         x = Activation("softmax")(x)
#         model = Model(inputs, x, name="mymodel")
#         return model


bs = 64  # batch size
resize_size = 96  # for training

import matplotlib.pyplot as plt
from PIL import Image
import Augmentor
import os

img_height = 96
img_width = 96
batch_size = 96
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
    print(X_val.shape)
    print(X_test.shape)
    # activate when using Machine Learning
    # X_train = X_train.reshape(7939, 27648)
    # X_val = X_val.reshape(1719, 27648)
    # X_test = X_test.reshape(1763, 27648)
    # X_train = X_train.astype(float) / 255
    # X_val = X_val.astype(float) / 255
    # X_test = X_test.astype(float) / 255

    # y_train = tf.keras.utils.to_categorical(y_train)
    # y_val = tf.keras.utils.to_categorical(y_val)
    # y_test = tf.keras.utils.to_categorical(y_test)
    print(X_train.shape, y_train[0])
    # (7939, 96, 96, 3)(7939, 4, 2)
    with open("bee_Wasp_XG.csv", "w", encoding="UTF-8") as f:
        """param_grid = [

            {'C': [0.1, 1, 10], 'degree': [2, 3], 'kernel': ['poly'], 'random_state': [1234]},

            {'C': [0.1, 1, 10], 'gamma': [0.0001, 0.001, 0.01],

             'kernel': ['rbf'],
             'random_state': [1234]}

        ]"""
        """
        ['bee' 'insect' 'other' 'wasp'] {'C': 10, 'gamma': 0.001, 'kernel': 'rbf', 'random_state': 1234} 0.6669619658465586
        """
        """
        bestSVC = SVC()
        print("start!")
        gsSVC = GridSearchCV(bestSVC, param_grid, n_jobs=4, verbose=2)
        gsSVC.fit(X_train, y_train)
        print(gsSVC.classes_, gsSVC.best_params_, gsSVC.best_score_)
        s = pickle.dumps(gsSVC)
        joblib.dump(s, 'SVCmodel.pkl')"""

        # filename = 'SVCmodel.pkl'
        #
        # with open(filename, 'rb') as file:
        #     loaded_model = pickle.loads(joblib.load(file))
        #
        #     y_test_pred = loaded_model.predict(X_test)
        #     accuracy = accuracy_score(y_test, y_test_pred)
        #
        #     ConMatrix = confusion_matrix(y_test, y_test_pred)
        #     ClassReport = classification_report(y_test, y_test_pred)
        #     print("Confusion Matrix Score:\n", ConMatrix)
        #     print("Classification Report: \n", ClassReport)
        #     f.write(
        #         "{}".format(loaded_model.__class__) + ',' + "{0:.4f}".format(accuracy) + ',' + "{}".format(
        #             loaded_model.best_params_) + '\n')
        # xgboost.XGBClassifier()
        #
        # XGB_best = xgboost.XGBClassifier()
        # XG_param = {"max_depth": [2, 4],
        #             'min_child_weight': [3, 6],
        #             'gamma': [0],
        #             'learning_rate': [0.5, 0.7],
        #             'random_state': [42]}
        # gsXG = GridSearchCV(XGB_best, XG_param, scoring='accuracy', verbose=2, n_jobs=4)
        # gsXG.fit(X_train, y_train)
        # t = pickle.dumps(gsXG)
        # joblib.dump(t, 'XGmodel.pkl')
        # XG_best = gsXG.best_estimator_
        # print(gsXG.classes_, gsXG.best_params_, gsXG.best_score_)
        # XG_pred = XG_best.predict(X_test)
        # accuracy = accuracy_score(y_test, XG_pred)
        # print("XGB 정확도: {0:.4f}".format(accuracy))
        # ConMatrix = confusion_matrix(y_test, XG_pred)
        # ClassReport = classification_report(y_test, XG_pred)
        # f.write(
        #     "{}".format(gsXG.__class__) + ',' + "{0:.4f}".format(accuracy) + ',' + "{}".format(gsXG.best_params_) + '\n')
        # print("Confusion Matrix Score:")
        # print(ConMatrix)
        # print("Classification Report:")
        # print(ClassReport)

    # for reproducibility
    import datetime

    if os.path.exists("./my_model_batch96_decay2000"):
        model = K.models.load_model("my_model_batch96_decay2000")
    else :
        model = ResNet.build(96, 96, 3, 4, (3, 4, 6), (64, 128, 256, 512), reg=0.0005)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=2000,
            decay_rate=0.96,
            staircase=True)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(X_train, y_train, epochs=30,
                            validation_data=(X_val, y_val), batch_size=batch_size, verbose=1)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        model.save("my_model_batch{}_decay2000".format(batch_size))
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
    suffix = datetime.datetime.now().strftime("%H%M%S")
    plt.savefig('loss_' + suffix + '.png')
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
    plt.savefig('acc_' + suffix + '.png')
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
    # arr0 = [1, 0, 0, 0]
    # arr1 = [0, 1, 0, 0]
    # arr2 = [0, 0, 1, 0]
    # arr3 = [0, 0, 0, 1]
    for i, j in zip(train.path, train.label):
        imgpath = dir + "/" + i
        # print(imgpath, j)
        img = Image.open(imgpath)
        img = img.convert("RGB")
        img = img.resize((img_height, img_height))

        data = np.asarray(img)
        trainImg.append(data)
        for _ in range(0, 10):
            tmp = AugImage(img)
            data1 = np.asarray(tmp)
            trainImg.append(data1)
            trainLabel.append(j)
        trainLabel.append(j)
        # if j == 'bee':
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         trainImg.append(data1)
        #         trainLabel.append(arr0)
        #     trainLabel.append(arr0)
        # elif j == 'wasp':
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         trainImg.append(data1)
        #         trainLabel.append(arr1)
        #     trainLabel.append(arr1)
        # elif j == 'insect':
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         trainImg.append(data1)
        #         trainLabel.append(arr2)
        #     trainLabel.append(arr2)
        # else:
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         trainImg.append(data1)
        #         trainLabel.append(arr3)
        #     trainLabel.append(arr3)

    for i, j in zip(val.path, val.label):
        imgpath = dir + "/" + i
        img = Image.open(imgpath)
        img = img.convert("RGB")
        img = img.resize((img_height, img_height))
        data = np.asarray(img)
        valImg.append(data)
        # if j == 'bee':
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         valImg.append(data1)
        #         valLabel.append(arr0)
        #     valLabel.append(arr0)
        # elif j == 'wasp':
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         valImg.append(data1)
        #         valLabel.append(arr1)
        #
        #     valLabel.append(arr1)
        # elif j == 'insect':
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         valImg.append(data1)
        #         valLabel.append(arr2)
        #     valLabel.append(arr2)
        # else:
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         valImg.append(data1)
        #         valLabel.append(arr3)
        #     valLabel.append(arr3)
        for _ in range(0, 10):
            tmp = AugImage(img)
            data1 = np.asarray(tmp)
            valImg.append(data1)
            valLabel.append(j)
        valLabel.append(j)

    for i, j in zip(test.path, test.label):
        imgpath = dir + "/" + i
        img = Image.open(imgpath)
        img = img.convert("RGB")
        img = img.resize((img_height, img_height))
        data = np.asarray(img)
        testImg.append(data)
        # if j == 'bee':
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         testImg.append(data1)
        #         testLabel.append(arr0)
        #     testLabel.append(arr0)
        # elif j == 'wasp':
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         testImg.append(data1)
        #         testLabel.append(arr1)
        #     testLabel.append(arr1)
        # elif j == 'insect':
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         testImg.append(data1)
        #         testLabel.append(arr2)
        #     testLabel.append(arr2)
        # else:
        #     for _ in range(0, 10):
        #         tmp = AugImage(img)
        #         data1 = np.asarray(tmp)
        #         testImg.append(data1)
        #         testLabel.append(arr3)
        #     testLabel.append(arr3)
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
