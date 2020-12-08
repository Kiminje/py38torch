import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

(x_train_all, y_train_all), (x_test, y_test) = imdb.load_data(skip_top=20, num_words=1000)
print(np.shape(x_train_all))
for i in range(len(x_train_all)):
    x_train_all[i] = [w for w in x_train_all[i] if w > 2]
print(np.shape(x_train_all))
np.random.seed(42)
random_index = np.random.permutation(25000)
x_train = x_train_all[random_index[:20000]]
y_train = y_train_all[random_index[:20000]]
x_val = x_train_all[random_index[20000:]]
y_val = y_train_all[random_index[20000:]]

maxlength=100
x_train_seq = sequence.pad_sequences(x_train, maxlen=maxlength)
x_val_seq = sequence.pad_sequences(x_val, maxlen=maxlength)

model_lstm = Sequential()
model_lstm.add(Embedding(1000,32))
model_lstm.add(LSTM(8))
model_lstm.add(Dense(1,activation='sigmoid'))
model_lstm.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model_lstm.summary()
history = model_lstm.fit(x_train_seq, y_train, epochs=10, batch_size=32, validation_data=(x_val_seq,y_val))
"""
Model: "sequential"
embedding : same as embedding rnn
LSTM :
    forget, input, output, temporary cell have same size of weight
    weight : W(h) + U(x) + b(bias) = 8 + 32 + 1
    # of params : 4 x (8 + 32 + 1) * 8(recurrent cell) 1312
dense : same as rnn

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 32)          32000     
_________________________________________________________________
lstm (LSTM)                  (None, 8)                 1312      
_________________________________________________________________
dense (Dense)                (None, 1)                 9         
=================================================================
Total params: 33,321
Trainable params: 33,321
Non-trainable params: 0
_________________________________________________________________
Train on 20000 samples, validate on 5000 samples
"""

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('ch09_06.png')
plt.clf()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.savefig('ch09_07.png')
plt.clf()

loss, accuracy = model_lstm.evaluate(x_val_seq, y_val, verbose=0)
print(accuracy)
"""
Epoch 10/10
625/625 [==============================] - 9s 14ms/step - loss: 0.2297 - accuracy: 0.9046 - val_loss: 0.4290 - val_accuracy: 0.8368
0.8367999792098999
"""
# Improve Strategy
"""
num_words : how many words you consider frequency orderly
skip_tops : top frequency words have less information(ex: 'the', 'a'..), how many words you skip
drop out & regularization : dense layer optimizing
embedding space dimension : x & U
cell dimension : h & W
"""
Dictionaries = [1000, 4000, 10000]
Skips = [0, 20, 40]
MaxLengths =[100, 250, 500, 1000, 2000]
Ratios = [0, 0.25, 0.5]
Cells = [8, 32, 128, 512]
EmbSpaces = [32, 64, 128, 256, 512]
regs = [0, 1e-5, 1e-4]
with open("imdb_result.csv", "w", encoding="UTF-8") as f:
    f.write("Dict size,Skip words,max length,drop out,cell size,embedding dim,regularization,test_loss,test_acc")
    for Dict in Dictionaries:
        for skip in Skips:
            for Max in MaxLengths:
                for drop in Ratios:
                    for cell in Cells:
                        for Emb in EmbSpaces:
                            for reg in regs:
                                (x_train_all, y_train_all), (x_test, y_test) = imdb.load_data(skip_top=skip, num_words=Dict)
                                print(np.shape(x_train_all))
                                for i in range(len(x_train_all)):
                                    x_train_all[i] = [w for w in x_train_all[i] if w > 2]
                                print(np.shape(x_train_all))
                                np.random.seed(42)
                                random_index = np.random.permutation(25000)
                                x_train = x_train_all[random_index[:20000]]
                                y_train = y_train_all[random_index[:20000]]
                                x_val = x_train_all[random_index[20000:]]
                                y_val = y_train_all[random_index[20000:]]
                                maxlength = Max
                                x_test = sequence.pad_sequences(x_test, maxlen=maxlength)
                                x_train_seq = sequence.pad_sequences(x_train, maxlen=maxlength)
                                x_val_seq = sequence.pad_sequences(x_val, maxlen=maxlength)


                                model_lstm = Sequential()
                                model_lstm.add(Embedding(Dict, Emb))
                                model_lstm.add(LSTM(cell))
                                model_lstm.add(Dropout(drop))
                                model_lstm.add(Dense(1, activation='sigmoid'))
                                model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                                model_lstm.summary()
                                history = model_lstm.fit(x_train_seq, y_train, epochs=10, batch_size=32,
                                                         validation_data=(x_val_seq, y_val))
                                print("training end!")
                                loss, accuracy = model_lstm.evaluate(x_test, y_test, verbose=0)
                                print(accuracy)
                                # "Dict size,Skip words,max length,drop out,cell size,embedding dim,regularization,test_loss,test_acc"
                                f.write("{}".format(Dict) + ',' + "{}".format(skip) + ',' + "{}".format(maxlength) + ',' + "{}".format(drop)
                                        + ',' + "{}".format(cell) + ',' + "{}".format(Emb) + ',' + "{}".format(reg)
                                        + ',' + "{0:.4f}".format(loss) + ',' + "{0:.4f}".format(accuracy) + '\n')


