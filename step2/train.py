from keras.models import Sequential
from keras.layers import Activation, Dense
from numpy import genfromtxt
import numpy as np
import h5py

X = genfromtxt('./some.csv',
               delimiter=',',
               skip_header=1,
               usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
               dtype=int)

Y = genfromtxt('./some.csv',
               delimiter=',',
               skip_header=1,
               usecols=(15),
               dtype=int)

model = Sequential()

# 15 -> 30
model.add(Dense(input_dim=15, output_dim=30))

# add tanh layer
model.add(Activation("tanh"))

# 30 -> 1
model.add(Dense(output_dim=1))

# add Sigmoid layer
model.add(Activation("sigmoid"))

# run compile
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# fit
model.fit(X, Y, nb_epoch=5000, batch_size=32)

# predict
results = model.predict_proba(np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
))

print(results)

# save model
model.save_weights('./learnedModel')
