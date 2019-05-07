import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

class GAN_DNN:
    def get_generator(self, ip_size):
        generator = Sequential()
        # The dimensions of noise = 100
        generator.add(Dense(units=256, input_dim=100, activation=LeakyReLU(0.2)))
        generator.add(Dense(units=512, activation=LeakyReLU(0.2)))
        generator.add(Dense(units=1024, activation=LeakyReLU(0.2)))
        generator.add(Dense(units=ip_size, activation='sigmoid'))
        
        return generator

    def get_descriminator(self, ip_size):
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim = ip_size, activation = LeakyReLU(alpha=0.2)))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(512, activation = LeakyReLU(alpha=0.2)))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(256, activation = LeakyReLU(alpha=0.2)))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(1, activation='sigmoid'))  
        
        return discriminator

    def plot_loss(self):
        plt.figure(figsize=(10,8))
        plt.plot(self.dloss, label="Discriminator loss")
        plt.plot(self.gloss, label="Generator loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def plot_generated(self, n_ex=10, dim=(1, 10), figsize=(12, 2)):
        noise = np.random.normal(0, 1, size=(n_ex, self.random_dim))
        generated_images = self.generator.predict(noise)
        generated_images = generated_images.reshape(n_ex, 28, 28)
    
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def __init__(self, ip_size, parameters):
        self.random_dim = 100
        self.gloss = []
        self.dloss = []
        self.generator = self.get_generator(ip_size)
        self.discriminator = self.get_descriminator(ip_size)
    
        opt = SGD(0.1, momentum=0.9)

        self.generator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.discriminator.trainable = False

        inputs = Input(shape=(self.random_dim, ))
        hidden = self.generator(inputs)
        output = self.discriminator(hidden)

        self.gan = Model(inputs, output)
        self.gan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def fit(self, x_train, y_train, batch_size, epochs = 100, verbose=0):
        batch_count = x_train.shape[0] // batch_size
        print ('Epochs:', epochs)
        print ('Batch size:', batch_size)
        print ('Batches per epoch:', batch_count)

        for epoch in range(1, epochs+1):
            print ('-'*15, 'Epoch %d' % epoch, '-'*15)

            for _ in tqdm(range(batch_count)):
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

                generated_images = self.generator.predict(noise)
                x = np.concatenate([image_batch, generated_images])

                y = np.zeros(2*batch_size)
                y[:batch_size] = 0.9

                self.discriminator.trainable = True
                dloss = self.discriminator.train_on_batch(x, y)

                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                y2 = np.ones(batch_size)
                self.discriminator.trainable = False
                gloss = self.gan.train_on_batch(noise, y2)

            self.dloss.append(dloss)
            self.gloss.append(gloss)

            if epoch == 1 or epoch % 10 == 0:
                self.plot_generated()

        self.plot_loss()

class GAN:
    def fit(self, x_train, y_train, batch_size, epochs = 15, verbose = 0):
        self.gan.fit(x_train, y_train, batch_size, epochs, verbose)

    def __init__(self, ip_size, parameters):
        if parameters['architecture'] == 'DNN':
            self.gan = GAN_DNN(ip_size, parameters)
        else:
            self.gan = GAN_CNN(ip_size, parameters)
