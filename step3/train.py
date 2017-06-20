from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

features = iris.data
targets = iris.target

model = Sequential()
model.add(Dense(12, input_dim=4))
model.add(Activation('relu'))
model.add(Dense(3, input_dim=12))
model.add(Activation('softmax'))

model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(features, targets, nb_epoch=1000, batch_size=5)

classes = model.predict_classes(
    np.array(
        [
            [5.2, 4.1, 1.5, 0.1],
            [5.9, 3., 5.1, 1.8]
        ]), batch_size=32, verbose=1
)

print(classes)
