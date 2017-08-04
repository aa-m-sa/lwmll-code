#!/usr/bin/env python3

# perceptron on scikit

import numpy as np
import matplotlib.pyplot as plt
# instead full MNIST, let's play with sklearn dataset
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron


np.random.seed(996) #predictable randomness

# this "ifmain" thing isn't actually needed, but I'd like to separate the "main function" like in C/C-like langs
# the following code is run only if you run 'python3 perceptron_scikit.py', not e.g. if you import the thing in other python code
if __name__ == "__main__":

    digits = load_digits()

    X_all = digits.data # n_samples x n_dim ( 1797 x 64)
    y_all = digits.target

    n_all = digits.data.shape[0]
    n_train = 100
    labels = np.unique(y_all)
    print(labels)

    # PML scales data with StandardScaler to 0 mean 1 std
    # ...looks like it does fancy in-place computations, might be useful for truly large datasets
    # but let's to do that by hand

    m = np.mean(X_all, 0)
    std = np.std(X_all, 0)
    print(X_all.shape)
    print(m.shape)
    print(std.shape)
    # I'd do this but oops, some of std elements are 0 (value same in all samples, no variance)
    #X_all = (X_all - m)/std
    print(m)
    print(std)

    # but then we can just ignore them, subtracting mean sets those elements to -
    std[std == 0] = 1
    X_all = (X_all - m)/std
    print(X_all.shape)

    # PML book uses helper function for that but it's easy to do by yourself
    permu = np.random.permutation(n_all)
    ind_train = permu[0:n_train]

    X_train = X_all[ind_train,:]
    y_train = y_all[ind_train]

    X_test = X_all[permu[n_train:],:]
    y_test = y_all[permu[n_train:]]

    ppn = Perceptron(n_iter = 1, eta0=0.01, random_state=0, warm_start=True)
    errs = []
    errs_tr = []
    for i in range(50):
        ppn.fit(X_train, y_train)

        ytr_pred = ppn.predict(X_train)

        y_pred = ppn.predict(X_test)
        err_n = (y_test != y_pred).sum()
        err_r = float(err_n) / y_pred.shape[0]
        print('Epoch {2} misclassifications: {0} rate {1}'.format(err_n, err_r,i))

        errs.append(err_r)

        errs_tr.append(float((y_train != ytr_pred).sum())/y_train.shape[0])

    plt.figure()
    ax1 = plt.subplot(211)
    plt.title('Scikit Perceptron,' +
              '(tr {0} samples), 1-vs-all multiclass n = 10 digits'.format(n_train))
    plt.plot(np.arange(50) + 1,errs_tr, 'b')
    plt.ylabel('train set error rate')
    ax1.set_ylim([0,1])
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(np.arange(50) + 1, errs, 'g')
    plt.ylabel('test set error rate')
    plt.xlabel('training iteration')
    ax2.set_ylim([0,1])
    #plt.savefig('percep_scikit_error_nostd')
    plt.savefig('percep_scikit_error_{0}'.format(n_train))
