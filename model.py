import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Model:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 1

        img_rows, img_cols = 28, 28
        self.input_shape = (img_rows, img_cols, 1)
        self.num_classes = 10

        self.model = self._build_model()
        

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model


    def train(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(x_test, y_test))

        score = self.model.evaluate(x_test, y_test, verbose=0)
        return score


    def classify(self, im):
        return self.model.predict(im).argmax()


    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)
