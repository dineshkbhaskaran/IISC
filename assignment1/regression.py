import numpy as np
from copy import deepcopy
from keras.datasets import mnist
from keras.utils import np_utils

def load_mnist_data(input_dim):
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, input_dim)
    x_train = x_train.astype('float32')
    x_train /= 255

    x_test = x_test.reshape(10000, input_dim)
    x_test = x_test.astype('float32')
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)

def one_hot_encoding(matrix, classes):
    return np.squeeze(np.eye(classes)[matrix.reshape(-1)])

def softmax(a):
    a_max = np.max(a, axis=-1, keepdims=True)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

def train_batch(x_train, y_train, weights, learning_rate=0.1):
    y = softmax(np.matmul(x_train, weights))

    gradient = np.matmul(x_train.T, y - y_train)/y_train.shape[0]
    weights -= learning_rate * gradient

    return weights

def validate(x_test, y_test, weights):
    y = softmax(np.matmul(x_test, weights))
    return np.mean(np.equal(np.argmax(y, axis=1), y_test))

def loss(x_test, y_test, weights):
    return -np.sum(y_test*np.log(softmax(np.matmul(x_test, weights))))/x_test.shape[0]

def train(x_train, y_train, classes, learning_rate=0.1, batch_size=128, max_iter=20):
    weights = np.zeros((np.size(x_train, 1), classes))

    weights_array = []
    for i in range(max_iter):
        for i in range(batch_size):
            if (i+1) * batch_size >= x_train.shape[0]:
                break
            x_new = x_train[i*batch_size:(i+1)*batch_size]
            y_new = y_train[i*batch_size:(i+1)*batch_size]

            weights = train_batch(x_new, y_new, weights, learning_rate)

        weights_array.append(deepcopy(weights))

    return weights_array

def print_data(data1, data2, data3, data4):
    print '[Training_loss] [Test_loss] [Train_Accuracy] [Test_Accuracy]'
    for i in range(len(data1)):
        print '%8.4f        %8.4f      %8.4f      %8.4f' % (data1[i], data2[i], data3[i], data4[i])
    
    print '\n'

def main():
    classes   = 10
    input_dim = 784

    (x_train, y_train), (x_test, y_test) = load_mnist_data(input_dim)

    learning_rate = [0.001, 0.01, 0.05, 0.1]
    batch_size = [1, 32, 128, 1024]

    y_train_enc = one_hot_encoding(y_train, classes)
    y_test_enc = one_hot_encoding(y_test, classes)

    for lr in learning_rate: 
        for bs in batch_size:
            print 'Training for lr = %f, bs = %d\n' % (lr, bs) 
            weights = train(x_train, y_train_enc, classes, lr, bs)

            loss_train = [loss(x_train, y_train_enc, weight) for weight in weights]
            loss_test = [loss(x_test, y_test_enc, weight) for weight in weights]
            accuracy_train = [validate(x_train, y_train, weight) for weight in weights]
            accuracy_test = [validate(x_test, y_test, weight) for weight in weights]

            print_data(loss_train, loss_test, accuracy_train, accuracy_test)

if __name__ == '__main__':
    main()

