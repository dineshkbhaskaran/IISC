import time
from keras import backend 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers import Dropout, BatchNormalization
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
    def __init__(self, nn_type, num_classes, parameters = {}):
        self.num_classes = num_classes
        self.nn_type = nn_type
        self.parameters = parameters

    def load_data(self, source):
        if source == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            self.validation_split = 10000.0/60000.0

        import pdb
        pdb.set_trace()
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)

    def get_model(self, normalize=False):
        model = Sequential()
        nlayers = 1

        if 'layers' in self.parameters.keys():
            nlayers = self.parameters['layers']

        dropout = [0.2] * nlayers
        if 'dropout' in self.parameters.keys():
            dropout = self.parameters['dropout']

        if self.nn_type == 'RNN':
            model.add(SimpleRNN(512, input_shape=self.x_train[0].shape, activation='relu'))
            model.add(Dropout(dropout[0]))
            if 'normalization' in self.parameters.keys():
                if self.parameters['normalization'] == True:
                    model.add(BatchNormalization())
        
        if self.nn_type == 'LSTM':
            for i in range(nlayers-1):
                model.add(LSTM(128, input_shape=self.x_train[0].shape, activation='relu', return_sequences=True))
                model.add(Dropout(dropout[i]))

            model.add(LSTM(128, input_shape=self.x_train[0].shape, activation='relu'))
            model.add(Dropout(dropout[nlayers-1]))
        #if self.nn_type == 'GAN':

        model.add(Dense(self.num_classes, activation='softmax'))
        model.summary()

        return model

    def fit(self, lr, bs):
        model = self.get_model()
        opt = SGD(lr, momentum=0.9)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        time_callback = time_history()
        history = model.fit(self.x_train, self.y_train, batch_size = bs, epochs = 15, verbose=1,
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

    def execute(self, batchsizes, learning_rates):
        self.load_data('mnist')

        for bs in batchsizes:
            for lr in learning_rates:
                stats = self.fit(lr, bs)
                print (stats)
        
    def cleanup(self):
        backend.clear_session()
