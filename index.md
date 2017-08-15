# Algorithms in this repo

(See [project page](https://github.com/aa-m-sa/lwmll-code) for readme etc.)

## What this repo contains?

A tour through Sebastian Raschka's book *Python Machine Learning* (Packt, 2015).
Let's review contents of an undergraduate ML course!

Most of algorithms are simple experiments on canned algorithms from [Scikit-Learn](http://scikit-learn.org/), but some of the more fun ones I implemented by hand, too.

## Table of Contents

1. [Perceptrons](https://aa-m-sa.github.io/lwmll-code/#perceptrons)
2. [Logistic Regression](https://aa-m-sa.github.io/lwmll-code/#logistic-regression)
3. [SVMs](https://aa-m-sa.github.io/lwmll-code/#svms)

## Perceptrons

### Files

* `perceptron_mnist.py` Simple 1-vs-1 binary perceptron classifier and tests on MNIST dataset.
* `adaline_mnist.py` [Adaline](https://en.wikipedia.org/wiki/ADALINE) with gradient descent, also MNIST.
* `perceptron_scikit.py` 1-vs-all digit classification with `sklearn` `Perceptron`. Uses smaller `sklearn` toy data set.

### Some results

Adaline, digits '5' vs '6' classification task error rate on training and test sets:

![](https://github.com/aa-m-sa/lwmll-code/raw/master/pics/ada_error_56.png)

Classification between '0' and '1' is easier:

![](https://github.com/aa-m-sa/lwmll-code/raw/master/pics/ada_error_10.png)

Scikit Perceptron, 1 vs all multiclass (all labels [0, ..., 1]) classification (different digit dataset here):

![](https://github.com/aa-m-sa/lwmll-code/raw/master/pics/percep_scikit_error_200.png)

`sklearn` implements perceptron as a special case of the more general linear SGD classifier, see details (like nice illustration how 1-vs-all classification works) [on sklearn docs](http://scikit-learn.org/stable/modules/sgd.html#classification).

## Logistic Regression

To keep results comparable, from now on I'm sticking to canned `digits` from `sklearn.datasets`.

### Files

* `logistic_scikit.py`

### Methods

* Logistic Regression with l1/l2 regularization
* solvers include
    * 'lfbgs' aka the Quasi-Newton workhorse, see [Wikipedia](https://en.wikipedia.org/wiki/L-BFGS)
    * 'SAGA', never heard of this particular strand of algorithms before. Combines [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) with averaging? [arXiv:1407.0202](https://arxiv.org/abs/1407.0202). See also [SAG](https://hal.inria.fr/hal-00860051/document)
    * also [liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/), also new to me. A C++ library with coordinate descent and Newton methods under the hood?

### Some results

Liblinear L2 surprisingly unsensitive to regularization constant:

![](https://github.com/aa-m-sa/lwmll-code/raw/master/pics/logistic_scikit_error_100_l2_liblinear.png)

L-BFGS a bit more like as I expected:

![](https://github.com/aa-m-sa/lwmll-code/raw/master/pics/logistic_scikit_error_100_l2_lbfgs.png)

Saga l2 (ridge regression) and l1 (LASSO):

![](https://github.com/aa-m-sa/lwmll-code/raw/master/pics/logistic_scikit_error_100_l2_saga.png)

![](https://github.com/aa-m-sa/lwmll-code/raw/master/pics/logistic_scikit_error_100_l1_saga.png)

Expected a more noticeable difference between these two, but OTOH we are fitting a fairly simple linear model and then evaluating performance of the decision boundaries, so maybe it does not make that much difference how strongly weights tend to 0 / 'sparse' they are.

(Also I always seem to learn new things by searching things I already 'know' simply searching relevant SE sites, [see](https://stats.stackexchange.com/questions/866/when-should-i-use-lasso-vs-ridge). TIL)

## SVMs

In progress...
