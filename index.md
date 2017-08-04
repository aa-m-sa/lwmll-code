# Algorithms in this repo

(See [project page](https://github.com/aa-m-sa/lwmll-code) for readme etc.)

## What this repo contains?

A tour through Sebastian Raschka's book *Python Machine Learning* (Packt, 2015).
Let's review contents of an undergraduate ML course!

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

Scikit Perceptron, 1 vs all multiclass (all labels [0, ..., 1]) classification:

![](https://github.com/aa-m-sa/lwmll-code/raw/master/pics/percep_scikit_error_200.png)

`sklearn` implements perceptron as a special case of the more general linear SGD classifier, see details (like nice illustration how 1-vs-all classification works) [on sklearn docs](http://scikit-learn.org/stable/modules/sgd.html#classification).
