#!/usr/bin/env python3

# logistic regression on scikit

import numpy as np
import matplotlib.pyplot as plt
# instead full MNIST, let's play with sklearn dataset
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression


np.random.seed(996) #predictable randomness

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
    print(m)
    print(std)

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

    errs = []
    errs_tr = []
    reg_cs = [0.0001, 0.01, 1.0, 100, 500, 1000, 2000]
    reg_type = 'l2'
    solver = 'lbfgs'
    multi_class = 'multinomial'
    for reg_c in reg_cs:
        lr = LogisticRegression(C=reg_c, penalty=reg_type, random_state = 0, multi_class = multi_class,
                                solver = solver)
        # PML book defines LR for two-class classification
        # with 'ovr', which is a default made explicit here, scikit.logisticRegression uses 1-vs-all
        # aka 1-vs-rest to adapt that to multiclass task
        # alternative: 'multinomial'
        lr.fit(X_train, y_train)

        ytr_pred = lr.predict(X_train)

        y_pred = lr.predict(X_test)
        err_n = (y_test != y_pred).sum()
        err_r = float(err_n) / y_pred.shape[0]
        print('C {2} misclassifications: {0} rate {1}'.format(err_n, err_r,reg_c))

        errs.append(err_r)

        errs_tr.append(float((y_train != ytr_pred).sum())/y_train.shape[0])

    plt.figure()
    ax1 = plt.subplot(211)
    plt.title('Scikit Logistic Reg,' +
              '(tr {0} samples), solver {1} , 1-vs-all, n = 10 digits'.format(n_train, solver))
    plt.plot(reg_cs,errs_tr, 'bx-')
    plt.ylabel('train set error rate')
    ax1.set_ylim([0,1])
    ax1.set_xscale('log')
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(reg_cs, errs, 'go-')
    plt.ylabel('test set error rate')
    plt.xlabel('{0} regularization penalty'.format(reg_type))
    ax2.set_ylim([0,1])
    ax2.set_xscale('log')
    #plt.savefig('percep_scikit_error_nostd')
    solver_printstr = solver
    if multi_class == 'multinomial':
        solver_printstr += 'multi'
    plt.savefig('logistic_scikit_error_{0}_{1}_{2}'.format(n_train, reg_type, solver_printstr))

