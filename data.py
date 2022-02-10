import jax
import jax.numpy as np
import numpy as onp
import torchvision
import torch
import os
from jax.config import config


# Enable float64 for accurate rank calculation
config.update("jax_enable_x64", True)


class CIFAR10:
    def __init__(self, n_train, n_test, classes=10, flat=False, key=None):
        """
        Implements data structure for CIFAR10, allowing to resize the images
        :param n_train:         int, number of training examples
        :param n_test:          int, number of test examples
        :param classes:         int, number of classes, one of '2' or '10'
        :param flat:            bool, flatten image to vector
        :param key:             key, used to generate randomness
        """
        self.n_train = n_train
        self.n_test = n_test
        self.flat = flat
        self.key = key
        self.classes = classes

        self.get_data()

    def get_data(self):
        # Load and store data
        dir_path = os.path.dirname(os.path.realpath(__file__))
        trainset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=False, download=True)

        # Sort the data
        self.x_train = torch.tensor(trainset.data[:, :, :]) / 255.0
        self.x_train = self.x_train.squeeze().numpy()
        self.y_train = np.array(trainset.targets)
        ind = np.argsort(self.y_train)
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

        self.x_train = np.concatenate(xs, axis=0)
        self.y_train = onp.expand_dims(np.concatenate(ys, axis=0), axis=1)

        self.x_test = torch.tensor(testset.data[:, :, :, :]) / 255.0
        self.x_test = self.x_test.squeeze().numpy()
        self.y_test = onp.expand_dims(testset.targets, axis=1)

        if self.classes == 2:
            self.n_test = min([self.n_test, 2000])
            self.y_train = 2 * (self.y_train - 1 / 2)

            where_test = self.y_test < 2
            indices_test = np.where(where_test > 0)
            self.x_test = self.x_test[indices_test[0], :, :, :]
            self.y_test = 2 * (self.y_test[indices_test[0]] - 1 / 2)

        else:
            self.x_test = self.x_test[:self.n_test, :, :, :]
            self.y_test = self.y_test[:self.n_test]
        if self.classes != 2:
            self.y_train = onp.expand_dims(self.y_train[:self.n_train], axis=1)

        if self.classes != 2:
            # If we use more than two classes, one-hot encode instead of -1, 1 targets
            self.y_train = jax.nn.one_hot(self.y_train, self.classes).squeeze()
            self.y_test = jax.nn.one_hot(self.y_test, self.classes).squeeze()

        if self.flat:
            # Flatten the data to vectors to use fully-connected architectures
            self.x_train = onp.reshape(self.x_train, (self.n_train, -1))
            self.x_test = onp.reshape(self.x_test, (self.n_test, -1))

        else:
            self.x_train = np.transpose(self.x_train, [0, 3, 2, 1])
            self.x_test = np.transpose(self.x_test, [0, 3, 2, 1])