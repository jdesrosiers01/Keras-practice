from keras.models import Sequential
from keras.layers import Activation, Dense
import numpy as np

# shape 2
X_list = [[0, 0], [1, 1], [0, 0], [1, 1], [1, 1], [1, 1]]

# shape 1
Y_list = [[0], [1], [0], [1], [1], [1]]

X = np.array(X_list)
Y = np.array(Y_list)

# sequential
model = Sequential()

# 2 -> 10
model.add(Dense(input_dim=2, output_dim=10))

# add tanh layer
model.add(Activation("tanh"))

# enable to add layer
# model.add(Dense(input_dim=10, output_dim=2))
# model.add(Activation("sigmoid"))

# 10 -> 1
model.add(Dense(output_dim=1))

# add Sigmoid layer
model.add(Activation("sigmoid"))

# compile
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# fit 3000 epoch
model.fit(X, Y, nb_epoch=1000, batch_size=32)

# predict
results = model.predict_proba(np.array(
    [
        [1, 1],
        [0, 0]
    ]
))

print(results)
