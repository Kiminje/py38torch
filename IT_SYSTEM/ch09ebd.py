import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding

(x_train_all, y_train_all), (x_test, y_test) = imdb.load_data(skip_top=20, num_words=1000)

for i in range(len(x_train_all)):
    x_train_all[i] = [w for w in x_train_all[i] if w > 2]

np.random.seed(42)
random_index = np.random.permutation(25000)
x_train = x_train_all[random_index[:20000]]
print(np.shape(x_train[1]))
y_train = y_train_all[random_index[:20000]]
x_val = x_train_all[random_index[20000:]]
y_val = y_train_all[random_index[20000:]]

maxlength=100
#   input data's dimension is 144, normalize it about 100
x_train_seq = sequence.pad_sequences(x_train, maxlen=maxlength)
x_val_seq = sequence.pad_sequences(x_val, maxlen=maxlength)

model_ebd = Sequential()
model_ebd.add(Embedding(1000,32))
model_ebd.add(SimpleRNN(8))
model_ebd.add(Dense(1,activation ='sigmoid'))

model_ebd.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_ebd.fit(x_train_seq, y_train, epochs=10, batch_size=32, validation_data=(x_val_seq, y_val))
model_ebd.summary()
"""
embedding : 
get input feature(1000) and produce output feature(32)
make dictionary for 1000 high occurrence words and embedded to embedding space (32)
every sequence's max length is 100
so embedding mapping function : 
    1000 -> 32 : 32000 parameters
output shape -> (batch size, max_length, output dimension)
simple_rnn : 
    recurrent weight : 8 x 8 = 64
    input weight : 8 x 32 = 256
 +  biases : 8 x 1 = 8
 ---------------------------------
 =  params : 328
dense :
weight(8) + bias(1) = 9
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 32)          32000     
_________________________________________________________________
simple_rnn (SimpleRNN)       (None, 8)                 328       
_________________________________________________________________
dense (Dense)                (None, 1)                 9         
=================================================================
Total params: 32,337
Trainable params: 32,337
Non-trainable params: 0
_________________________________________________________________
"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('ch09_04.png')
plt.clf()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.savefig('ch09_05.png')
plt.clf()

loss, accuracy = model_ebd.evaluate(x_val_seq, y_val, verbose=0)
print(accuracy)

