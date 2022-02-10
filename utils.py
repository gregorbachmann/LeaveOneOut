import numpy as np
import jax.numpy as jnp

from data import CIFAR10, MNIST


def pseudo_divide(a, b, rank):
    """
    Performs a/b for last 'rank' entries and replaces the first n-'rank' entries with zeros
    :param a:       array of shape n
    :param b:       array of shape n
    :param rank:    int, threshold after which to zero
    :return:        array of shape n
    """
    n = b.shape[0]
    b = np.concatenate([np.zeros_like(b[rank:]), b[n-rank:]])
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


def accuracy(preds, targs):
    """
    Calculates accuracy of prediction 'preds' with respect to ground-truth 'targs'
    :param preds:   array of shape n x K
    :param targs:   array of shape n x K
    :return:        float
    """
    num_samples, num_out = preds.shape
    if num_out == 1:
        acc = jnp.mean(jnp.sign(preds) == targs)
    else:
        acc = 1 / num_samples * jnp.sum(jnp.argmax(preds, axis=1) == jnp.argmax(targs, axis=1), axis=0)

    return acc


def mse_loss(preds, targs):
    """
    Calculates the mean squared loss of prediction 'preds' with respect to ground-truth 'targs'
    :param preds:   array of shape n x K
    :param targs:   array of shape n x K
    :return:        float
    """
    num_samples, num_out = preds.shape
    if num_out == 1:
        loss = jnp.mean((preds - targs)**2)
    else:
        loss = 1 / num_samples * jnp.sum((preds - targs)**2)

    return loss


def get_dataset(name, n_train, n_test, classes, flat):
    """
    Returns the dataset according to 'name'
    :param name:        str, encoding name of dataset, one of 'CIFAR', 'MNIST'
    :param n_train:     int, number of training examples
    :param n_test:      int, number of test examples
    :param classes:     int, number of classes, one of 2 or 10
    :param flat:        bool, if true input arrays are flattened
    :return:            class of type dataset
    """
    if name == 'CIFAR':
        return CIFAR10(n_train, n_test, classes, flat)
    if name == 'MNIST':
        return MNIST(n_train, n_test, classes, flat)