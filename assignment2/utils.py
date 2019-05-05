import time
from keras import backend 
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv3D, MaxPooling3D
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.utils import to_categorical

class time_history(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class neural_networks():
    def __init__(self, nn_type, num_classes=10):
        self.num_classes = 10
        self.nn_type = nn_type

    def load_data(self, source):
        if source == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            self.validation_split = 10000.0/60000.0
        elif source == 'cifar':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            self.validation_split = 8000.0/50000.0

        # Data shaping is for assignment purpose only.
        if self.nn_type == 'CNN_2D':
            x_train = x_train.reshape(60000, x_train.shape[1], x_train.shape[2], 1)
            x_test = x_test.reshape(10000, x_train.shape[1], x_train.shape[2], 1)
        elif self.nn_type == 'DNN':
            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
        elif self.nn_type == 'CNN_3D':
            # This must be cifar data
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1, x_train.shape[3])
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1, x_test.shape[3])

        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)

    def get_model(self):
        model = Sequential()

        if self.nn_type == 'CNN_2D':
            model.add(Conv2D(128, kernel_size=3, padding="same", activation='relu', 
                input_shape=(28, 28, 1)))
            model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Conv2D(128, kernel_size=3, padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
        elif self.nn_type == 'DNN':
            model.add(Dense(512, activation='relu', input_shape=(784,)))
            model.add(Dense(512, activation='relu'))
        elif self.nn_type == 'CNN_3D':
            model.add(Conv3D(32, kernel_size=3, padding="same", activation='relu', 
                input_shape = self.x_train.shape[1:]))
            model.add(MaxPooling3D(pool_size=(2,2,1)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))

        model.add(Dense(self.num_classes, activation='softmax'))
        model.summary()

        return model
    
    def fit(self, lr, bs):
        model = self.get_model()
        opt = SGD(lr)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        time_callback = time_history()
        history = model.fit(self.x_train, self.y_train, batch_size = bs, epochs = 15, verbose=2,
                validation_split=self.validation_split, callbacks=[time_callback])
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
 
        stats = {}
        stats['lr'] = lr
        stats['bs'] = bs
        stats['val_acc'] = history.history['val_acc'][-1]
        stats['test_acc'] = score[1]
        stats['time'] = time_callback.times[-1]
        
        backend.clear_session()

        return stats

    def cleanup(self):
        backend.clear_session()
