import tensorflow_datasets as tfds
import tensorflow as tf

import os
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import numpy as np

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
import numpy as np

print(tf.__version__)
img_height = 64
img_width = 64
batch_size = 128
import pathlib

train_dir = "/home/inje/py38torch/Machine_Learning/cropped/train/"
test_dir = "/home/inje/py38torch/Machine_Learning/cropped/test/"


def get_label_from_path(path):
    return path.split('/')[-1]


def get_truncate_from_path(path):
    return path.split('-')[-1]


if not os.path.exists("./data/Dob_save.npz"):
    data_list = glob('/home/inje/py38torch/Machine_Learning/cropped/train/*')
    path = data_list
    nb_classes = len(path)
    Label = []
    Cat_path = []
    for i in path:
        tmp = get_label_from_path(i)
        Cat_path.append(tmp)
        Label.append(get_truncate_from_path(tmp))


    X = []
    Y = []

    for idx, cat in enumerate(Cat_path):
        label = [0 for i in range(nb_classes)]
        label[idx] = 1

        image_dir = train_dir + "/" + cat
        files = glob(image_dir + "/*.jpg")
        for i, j in enumerate(files):
            img = Image.open(j)
            img = img.convert("RGB")
            img = img.resize((img_height, img_height))
            data = np.asarray(img)

            X.append(data)
            Y.append(label)
    np.savez("./data/Dog_save.npz", x=X, y=Y)

    for i in Label:
        print(i)
    print(len(Label))
else:
    X, y = np.load("./data/Dob_save.npz")
    X_train, X_test, y_train, y_test = train_test_split(X, y, )



# print(data_list[0], get_label_from_path(data_list[0]))
#
# image = np.array(Image.open(path))
#
#
# def read_image(path):
#     image = np.array(Image.open(path))
#     # Channel 1을 살려주기 위해 reshape 해줌
#     return image.reshape(image.shape[0], image.shape[1], 1)
#
#
# class_name =get_label_from_path(path)
# print(class_name)
# label_name_list = []
#
# for path in data_list:
#     label_name_list.append(get_label_from_path(path))
#
#
# unique_label_names = np.unique(label_name_list)
#
# def onehot_encode_label(path):
#     onehot_label = unique_label_names == get_label_from_path(path)
#     onehot_label = onehot_label.astype(np.uint8)
#     return onehot_label
#
# print(onehot_encode_label(path))
#
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     train_dir,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(img_height, img_width),
#     batch_size=batch_size)
#
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     train_dir,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=(img_height, img_width),
#     batch_size=batch_size)
#
# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     test_dir,
#     seed=123,
#     image_size=(img_height, img_width),
#     batch_size=batch_size)
#
# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


# class_names = train_ds.
# print(class_names, class_names.shape)
# class_list = []
# for i in (0, 120):
#     class_list.append(class_names[i])
#     print(i)



# builder = tfds.ImageFolder(train_dir)
# print(builder.info)  # num examples, labels... are automatically calculated
# train_ds = builder.as_dataset(split='train', shuffle_files=True)
# tfds.show_examples(train_ds, builder.info)
# # class_names = train_ds.class_names
# builder = tfds.ImageFolder(train_dir)
# print(builder.info)  # num examples, labels... are automatically calculated
# test_ds = builder.as_dataset(split='test', shuffle_files=True)
# tfds.show_examples(test_ds, builder.info)
# # class_names = train_ds.class_names

# (x, y) = train_ds
# print(x.shape, y.shape)
# print(train_ds.output_shapes)
# print(class_names)
# print(np.shape(train_ds))
#
# AUTOTUNE = tf.data.experimental.AUTOTUNE
#
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# num_classes = 120
#
# model = tf.keras.Sequential()
# ChNums = [128]
# layers = [2]
# Ratios = [0.25]
# i = 1e-5
# j = 1e-5
# for Ch in ChNums:
#     for Layer in layers:
#         Split = Layer // 2
#         for ratio in Ratios:
#             conv2 = tf.keras.Sequential()
#             conv2.add(Conv2D(Ch, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
#             for _ in range(Split):
#                 conv2.add(Conv2D(Ch, (3, 3), activation='relu', padding='same'))
#             conv2.add(MaxPooling2D((2, 2)))
#             for _ in range(Split):
#                 conv2.add(Conv2D(Ch, (3, 3), activation='relu', padding='same'))
#             conv2.add(MaxPooling2D((2, 2)))
#             conv2.add(Flatten())
#             conv2.add(Dropout(ratio))
#             conv2.add(Dense(256, activation='relu'))
#             conv2.add(Dropout(ratio))
#             conv2.add(Dense(120, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=i, l2=j)))
#
#             conv2.summary()
#
#             conv2.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#                           metrics=['accuracy'])
#             history = conv2.fit(train_ds, epochs=30, validation_data=val_ds)
#             print("training end!")
#             loss, accuracy = conv2.evaluate(test_ds, verbose=1)
#             print(accuracy)
#
#             plt.plot(history.history['loss'])
#             plt.plot(history.history['val_loss'])
#             plt.ylabel('loss')
#             plt.xlabel('epoch')
#             plt.legend(['train_loss', 'val_loss'])
#             plt.grid(b=True, which='major', color='#666666', linestyle='-')
#             plt.minorticks_on()
#             plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#             plt.title("<LOSS: {:0.3f}> Layer {}, Channel {}, Ratio: {}".format(loss, Layer, Ch, ratio))
#             plt.savefig('ch08_loss{}_{}_{}.png'.format(Layer, Ch, ratio))
#             plt.clf()
#
#             plt.plot(history.history['accuracy'])
#             plt.plot(history.history['val_accuracy'])
#             plt.ylabel('accuracy')
#             plt.xlabel('epoch')
#             plt.legend(['train_accuracy', 'val_accuracy'])
#             plt.grid(b=True, which='major', color='#666666', linestyle='-')
#             plt.minorticks_on()
#             plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#             plt.title("<Accuracy: {:0.3f} > Layer {}, Channel {}, Ratio: {}".format(accuracy, Layer, Ch, ratio))
#             plt.savefig('ch08_acc{}_{}_{}.png'.format(Layer, Ch, ratio))
#             plt.clf()
#
# model.compile(
#   optimizer='adam',
#   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#   metrics=['accuracy'])
#
# model.summary()
# model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=3
# )
