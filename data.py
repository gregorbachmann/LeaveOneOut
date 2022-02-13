import jax
import jax.numpy as jnp
import numpy as np
import torchvision
import torch
import os
from jax.config import config


# Enable float64 for accurate rank calculation
config.update("jax_enable_x64", True)


class CIFAR10:
    def __init__(self, n_train, n_test, classes=10, flat=False, download=False):
        """
        Implements data structure for CIFAR10
        :param n_train:         int, number of training examples
        :param n_test:          int, number of test examples
        :param classes:         int, number of classes, one of '2' or '10'
        :param flat:            bool, flatten image to vector
        :param download:    bool, if true download the dataset
        """
        self.n_train = n_train
        self.n_test = n_test
        self.flat = flat
        self.classes = classes
        self.download = download

        self.get_data()

    def get_data(self):
        # Load and store data
        dir_path = os.path.dirname(os.path.realpath(__file__))
        trainset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=True, download=self.download)
        testset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=False, download=self.download)

        # Sort the data
        self.x_train = trainset.data[:, :, :] / 255.0
        self.x_train = self.x_train.squeeze()
        self.y_train = jnp.array(trainset.targets)
        ind = jnp.argsort(self.y_train)
        self.x_train = self.x_train[ind, ...]
        self.y_train = self.y_train[ind]

        xs = []
        ys = []
        if self.classes == 2:
            self.n_train = min([self.n_train, 10000])

        for k in range(self.classes):
            if self.classes == 2 and k == 2:
                break
            per_class = 5000
            per_class_n = self.n_train // self.classes
            xs.append(self.x_train[k * per_class: k * per_class + per_class_n])
            ys.append(self.y_train[k * per_class: k * per_class + per_class_n])

        self.x_train = jnp.concatenate(xs, axis=0)
        self.y_train = np.expand_dims(jnp.concatenate(ys, axis=0), axis=1)

        self.x_test = testset.data[:, :, :, :] / 255.0
        self.x_test = self.x_test.squeeze()
        self.y_test = np.expand_dims(testset.targets, axis=1)

        if self.classes == 2:
            self.n_test = min([self.n_test, 2000])
            self.y_train = 2 * (self.y_train - 1 / 2)

            where_test = self.y_test < 2
            indices_test = jnp.where(where_test > 0)
            self.x_test = self.x_test[indices_test[0], :, :, :]
            self.y_test = 2 * (self.y_test[indices_test[0]] - 1 / 2)

        else:
            self.x_test = self.x_test[:self.n_test, :, :, :]
            self.y_test = self.y_test[:self.n_test]
        if self.classes != 2:
            self.y_train = np.expand_dims(self.y_train[:self.n_train], axis=1)

        if self.classes != 2:
            # If we use more than two classes, one-hot encode instead of -1, 1 targets
            self.y_train = jax.nn.one_hot(self.y_train, self.classes).squeeze()
            self.y_test = jax.nn.one_hot(self.y_test, self.classes).squeeze()

        if self.flat:
            # Flatten the data to vectors to use fully-connected architectures
            self.x_train = np.reshape(self.x_train, (self.n_train, -1))
            self.x_test = np.reshape(self.x_test, (self.n_test, -1))

        else:
            self.x_train = jnp.transpose(self.x_train, [0, 3, 2, 1])
            self.x_test = jnp.transpose(self.x_test, [0, 3, 2, 1])


class MNIST:
    def __init__(self, n_train, n_test, classes=10, flat=False, download=False):
        """
        Implements data structure for MNIST
        :param n_train:         int, number of training examples
        :param n_test:          int, number of test examples
        :param classes:         int, number of classes, one of '2' or '10'
        :param flat:            bool, flatten image to vector
        :param download:    bool, if true download the dataset
        """
        self.n_train = n_train
        self.n_test = n_test
        self.flat = flat
        self.classes = classes
        self.download = download

        self.get_data()

    def get_data(self):
        # Load and store data
        dir_path = os.path.dirname(os.path.realpath(__file__))
        trainset = torchvision.datasets.MNIST(root=dir_path + '/data', train=True, download=self.download)
        testset = torchvision.datasets.MNIST(root=dir_path + '/data', train=False, download=self.download)

        # Sort the data
        self.x_train = trainset.train_data / 255.0
        self.x_train = self.x_train.squeeze().numpy()
        self.y_train = jnp.array(trainset.train_labels)
        counts = jnp.sum(jax.nn.one_hot(self.y_train, self.classes), axis=0)
        ind = jnp.argsort(self.y_train)
        self.x_train = self.x_train[ind, ...]
        self.y_train = self.y_train[ind]

        xs = []
        ys = []
        if self.classes == 2:
            self.n_train = min([self.n_train, 10000])

        per_class = 0
        for k in range(self.classes):
            if self.classes == 2 and k == 2:
                break
            per_class_n = self.n_train // self.classes
            xs.append(self.x_train[per_class: per_class + per_class_n])
            ys.append(self.y_train[per_class: per_class + per_class_n])
            per_class += int(counts[k])

        self.x_train = jnp.concatenate(xs, axis=0)
        self.y_train = np.expand_dims(jnp.concatenate(ys, axis=0), axis=1)

        self.x_test = testset.train_data / 255.0
        self.x_test = self.x_test.squeeze().numpy()
        self.y_test = np.expand_dims(testset.train_labels, axis=1)

        if self.classes == 2:
            self.n_test = min([self.n_test, 2115])
            self.y_train = 2 * (self.y_train - 1 / 2)

            where_test = self.y_test < 2
            indices_test = jnp.where(where_test > 0)
            self.x_test = self.x_test[indices_test[0], :, :]
            self.y_test = 2 * (self.y_test[indices_test[0]] - 1 / 2)

        else:
            self.x_test = self.x_test[:self.n_test, :, :]
            self.y_test = self.y_test[:self.n_test]
        if self.classes != 2:
            self.y_train = np.expand_dims(self.y_train[:self.n_train], axis=1)

        if self.classes != 2:
            # If we use more than two classes, one-hot encode instead of -1, 1 targets
            self.y_train = jax.nn.one_hot(self.y_train, self.classes).squeeze()
            self.y_test = jax.nn.one_hot(self.y_test, self.classes).squeeze()

        if self.flat:
            # Flatten the data to vectors to use fully-connected architectures
            self.x_train = np.reshape(self.x_train, (self.n_train, -1))
            self.x_test = np.reshape(self.x_test, (self.n_test, -1))

        else:
            self.x_train = jnp.transpose(self.x_train, [0, 3, 2, 1])
            self.x_test = jnp.transpose(self.x_test, [0, 3, 2, 1])
