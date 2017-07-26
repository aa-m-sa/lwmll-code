#!/usr/bin/env python3
# demonstrates a adaline perceptron classifier on MNIST dataset

# I'm lazy and use preprocessor code for MNIST dataset https://github.com/sorki/python-mnist
from mnist import MNIST

import numpy as np

import matplotlib.pyplot as plt
import warnings

np.random.seed(996) #predictable randomness

np.seterr(all='warn')

def train_adaline(data, values, n_iter=100, step=0.01, batchsize = 1):
    # batchsize == 1: stochastic gradient descent
    # batchsize == dn: full dataset
    # batches are selected in deterministic fashion
    dn = len(data)
    ddim = len(data[0])

    order = np.random.permutation(dn)
    data = data[order]
    values = values[order]

    w_all = np.zeros([n_iter + 1, ddim + 1]) # matrix; we store the whole history of w

    for n in range(n_iter):
        indices = batchsize * n + np.arange(batchsize)
        x = data[indices % dn] # batchsize x ddim
        y = values[indices % dn]
        # basic gradient descent with fixed stepsize (step)

        wtx = w_all[n, 0] + np.dot(x, w_all[n, 1:]) # prediction = wtx

        w_all[n+1, 0] = w_all[n, 0] + step * np.sum(y - wtx)
        w_all[n+1, 1:] = w_all[n,1:] + step * np.dot(x.T, y - wtx)

    return w_all

def test_adaline(w, data, values):
    # matrix operation would be fast but meh
    dn = len(data)
    err = 0.0
    for n in range(dn):
        x = data[n]
        y = values[n]
        wtx = w[0] + np.dot(x, w[1:])
        pred = np.sign(wtx)
        err += np.abs(pred - y) > 1e-8
    return err/dn # percentage of errors in the test set


def prepare_set(images, labels, a, b, subset=None):
    images_0 = [i for (i,l) in zip(images, labels) if l == a]
    d0 = len(images_0)
    images_1 = [i for (i,l) in zip(images, labels) if l == b]
    d1 = len(images_1)

    if subset:
        # take only part of the images
        images_0 = images_0[0:min([d0, subset])]
        images_1 = images_1[0:min([d1, subset])]

    images_set = images_0 + images_1 # concatenate
    values_set = np.hstack([np.ones(len(images_0)), -np.ones(len(images_1))])
    return np.array(images_set), np.array(values_set)

if __name__ == "__main__":
    mndata = MNIST('./mnist/') # your data file location
    images_raw, labels = mndata.load_training()
    print('loaded ', str(len(images_raw)), 'train images')
    images_raw_test, labels_test = mndata.load_testing()
    print('loaded ', str(len(images_raw_test)), 'test images')
    # images_raw is a list of lists
    # convert to numpy ndarrays and normalize to floats [-1.0, 1.0]

    images = [-1.0 + 2*np.array(i)/255 for i in images_raw]
    images_test = [-1.0 + 2*np.array(i)/255 for i in images_raw_test]

    # digits to classify
    for (a,b) in [(1,0), (5,8), (5,6)]:
        #images_train, values_train = prepare_set(images, labels, a, b, subset=200)
        images_train, values_train = prepare_set(images, labels, a, b, subset=1000)
        print(images_train.shape)
        images_test, values_test   = prepare_set(images_test, labels_test, a, b, subset=500)
        print(images_test.shape)

        # sneak a peek
        #plt.figure()
        #plt.imshow(np.reshape(images_0[0], [28, 28]))
        #plt.show()
        #plt.imshow(np.reshape(images_1[0], [28, 28]))
        #plt.show()

        print('starting training')
        n_iter = 400
        #step = 0.01
        # batch = 10
        step = 0.00001
        #batch = 200
        batch = 50
        w_all = train_adaline(images_train, values_train,
                                 n_iter = n_iter, step = step, batchsize = batch)
        print('done: n_iter', str(n_iter))

        # training error?
        print('evaluate error')
        err_train = np.zeros(n_iter)
        err_test = np.zeros(n_iter)
        for n in range(n_iter):
            err_train[n] = test_adaline(w_all[n,:], images_train, values_train)
            err_test[n]  = test_adaline(w_all[n,:], images_test, values_test)

        # plotting
        plt.figure()
        ax1 = plt.subplot(211)
        plt.plot(np.arange(n_iter), err_train, 'b')
        plt.ylabel('train error')
        plt.title('Classification '+str(a)+' vs '+str(b)+' : tr size = ' + str(images_train.shape[0]) +' batch = ' + str(batch) + ' step = '+str(step) )
        ax1.set_ylim([0,1])
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(np.arange(n_iter), err_test, 'g')
        plt.xlabel('adaline train iteration')
        plt.ylabel('test error')
        ax2.set_ylim([0,1])
        #plt.show()
        plt.savefig('ada_error_'+str(a)+str(b)+'.png')

