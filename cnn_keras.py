from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras import Sequential
import keras


class LeNet5:
    def __init__(self, input_size=(32, 32, 3), classes=2, learning_rate=1e-3):
        self.model = Sequential()
        self.model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_size))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=learning_rate),
                           metrics=['accuracy'])

    def fit(self, data, targets, batch_size=None):
        self.model.fit(data, targets, batch_size=batch_size, epochs=1, verbose=0)

    def accuracy(self, data, targets):
        return self.model.test_on_batch(data, targets)[1]

    def loss(self, data, targets):
        return self.model.test_on_batch(data, targets)[0]


class AlexNet:
    def __init__(self, input_size=(227, 227, 3), classes=1000, learning_rate=1e-3):
        self.model = Sequential()

        self.model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_size))

        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(Conv2D(256, (5, 5), strides=(1, 1), padding=2, activation='relu'))

        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(Conv2D(384, (3, 3), strides=(1, 1), padding=1, activation='relu'))

        self.model.add(Conv2D(384, (3, 3), strides=(1, 1), padding=1, activation='relu'))

        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding=1, activation='relu'))

        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dropout(rate=0.5))

        self.model.add(Dense(4096, activation='relu'))

        self.model.add(Dropout(rate=0.5))

        self.model.add(Dense(4096, activation='relu'))

        self.model.add(Dense(classes, activation='relu'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.RMSprop(lr=learning_rate, decay=0.0005),
                           metrics=['accuracy'])

    def fit(self, data, targets, batch_size=None):
        self.model.fit(data, targets, batch_size=batch_size, epochs=1, verbose=0)

    def accuracy(self, data, targets):
        return self.model.test_on_batch(data, targets)[1]

    def loss(self, data, targets):
        return self.model.test_on_batch(data, targets)[0]


class VGG19:
    def __init__(self, input_size=(224, 224, 3), classes=1000, learning_rate=1e-3):
        self.model = Sequential()

        self.model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_size))

        self.model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_size))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), activation='relu', input_shape=input_size))

        self.model.add(Conv2D(128, (3, 3), activation='relu', input_shape=input_size))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), activation='relu', input_shape=input_size))

        self.model.add(Conv2D(256, (3, 3), activation='relu', input_shape=input_size))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), activation='relu', input_shape=input_size))

        self.model.add(Conv2D(512, (3, 3), activation='relu', input_shape=input_size))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), activation='relu', input_shape=input_size))

        self.model.add(Conv2D(512, (3, 3), activation='relu', input_shape=input_size))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dense(4096, activation='relu'))

        self.model.add(Dense(4096, activation='relu'))

        self.model.add(Dense(classes, activation='relu'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.RMSprop(lr=learning_rate, decay=0.0005),
                           metrics=['accuracy'])

    def fit(self, data, targets, batch_size=None):
        self.model.fit(data, targets, batch_size=batch_size, epochs=1, verbose=0)

    def accuracy(self, data, targets):
        return self.model.test_on_batch(data, targets)[1]

    def loss(self, data, targets):
        return self.model.test_on_batch(data, targets)[0]
