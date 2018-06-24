from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras import Sequential
import keras


class LeNet5:
    def __init__(self, input_size=(32, 32, 3), classes=2, learning_rate=1e-3):
        self.model = Sequential()
        self.model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_size))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (5, 5), activation='tanh'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(classes, activation='tanh'))

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=learning_rate),
                           metrics=['accuracy'])

    def fit(self, data, targets):
        self.model.fit(data, targets, batch_size=data.shape[0], epochs=1, verbose=0)

    def loss(self, data, targets):
        return self.model.test_on_batch(data, targets)[0]
