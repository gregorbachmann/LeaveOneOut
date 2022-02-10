import jax.numpy as jnp
from jax import random
from neural_tangents import stax
from jax.config import config

from utils import pseudo_divide


# Enable float64 for accurate rank calculation
config.update("jax_enable_x64", True)


class Kernel:
    """Implements the base class for kernels"""
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.y_train = None
        self.classes = None
        self.dim = None
        self.n_train = None

    def fit(self, x_train, y_train):
        """
        Fits the kernel to the training data
        :param x_train:  Tensor of shape n_train x dim, input train features
        :param y_train:  Tensor of shape n_train x classes, train targets
        :param classes:  int, number of classes in dataset
        """
        self.x_train = x_train
        self.y_train = y_train
        self.n_train, self.dim = self.x_train.shape
        self.classes = self.y_train.shape[1]

        # Calculate the training kernel matrix and perform eigendecomposition
        self.K_train_train = self.compute_kernel(x_train, x_train)
        self.eigs_vals, self.eig_vecs = jnp.linalg.eigh(self.K_train_train)
        # Calculate rank and form pseudo inverse in case of rank deficiency
        tol = self.K_train_train.max() * self.n_train * jnp.finfo(self.K_train_train.dtype).eps
        self.rank = jnp.sum(self.eigs_vals > tol)
        diag_inv = jnp.diag(pseudo_divide(jnp.ones_like(self.eigs_vals), self.eigs_vals, self.rank))
        pseudo_inv = self.eig_vecs @ diag_inv @ self.eig_vecs.T
        # Form the predictive vector
        self.v = pseudo_inv @ self.y_train
        self.diag = jnp.diag(pseudo_inv)

    def compute_kernel(self, x1, x2):
        """
        Implements the kernel computation
        :param x1: Tensor of shape n1 x dim
        :param x2: Tensor of shape n2 x dim
        :return: Tensor of shape n1 x n2
        """
        raise NotImplemented

    def predict(self, x, mode):
        """
        Implements the predictive function based on kernel
        :param x:       Tensor of shape n x dim
        :param mode:    str, one of 'train' or else
        :return:        Tensor of shape n x K
        """
        if mode == 'train':
            return self.K_train_train @ self.v
        else:
            K_train_x = self.compute_kernel(x, self.x_train)
            return K_train_x @ self.v

    def leave_one_out(self):
        """Implements the leave-one-out loss and accuracy computation"""
        if self.rank == self.n_train:
            delta = self.v / jnp.expand_dims(self.diag, axis=1)
        else:
            eig_vecs_partial = self.eig_vecs[:, :(self.n_train - self.rank)]
            a = eig_vecs_partial @ eig_vecs_partial.T
            delta = a @ self.y_train / jnp.expand_dims(jnp.diag(a), axis=1)

        loo_loss = 1/self.n_train * jnp.sum(delta ** 2)

        if self.classes == 2:
            loo_acc = 1 / self.n_train * jnp.sum(self.y_train * delta < 1)

        else:
            true_class = jnp.argmax(self.y_train, axis=1)
            delta_class = jnp.argmax(self.y_train - delta, axis=1)
            loo_acc = 1 / self.n_train * jnp.sum(true_class == delta_class)

        return loo_loss, loo_acc


def ntk_kernel(depth=3):
    """
    Implements the NTK kernel of a fully-connected network
    :param depth:   int, depth of network
    :return: kernel function
    """
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512), stax.Relu(),
        stax.Dense(512), stax.Relu(),
        stax.Dense(512), stax.Relu(),
        stax.Dense(1)
    )

    return lambda x1, x2: kernel_fn(x1, x2, 'ntk')


key = random.PRNGKey(2)
_v = random.normal(key=key, shape=(10000, 10000))


def random_feature_kernel(depth, dim, width):
    """
    Implements the random feature kernel
    :param depth:   int, depth of network
    :param dim:     int, dimensionality of input features
    :param width:   int, width of network
    :return:        kernel function
    """
    v = _v[:dim, :width]
    def feature(x): return jnp.abs(x @ v)

    return lambda x1, x2: feature(x1) @ feature(x2).T


class NTK(Kernel):
    """Implements the NTK kernel"""
    def __init__(self, depth):
        """
        :param depth:    int, depth of network
        """
        self.depth = depth
        super().__init__()

    def compute_kernel(self, x1, x2):
        return ntk_kernel(depth=self.depth)(x1, x2)


class RF(Kernel):
    """Implements the random feature kernels"""
    def __init__(self, depth, width):
        """
        :param depth:   int, depth of network
        :param width:   int, width of network
        """
        self.depth = depth
        self.width = width
        super().__init__()

    def compute_kernel(self, x1, x2):
        return random_feature_kernel(depth=self.depth, dim=self.dim, width=self.width)(x1, x2)


unit_test = False

if unit_test:
    from data import CIFAR10
    n_test = 10000
    data = CIFAR10(n_train=10000, n_test=n_test, classes=10, flat=True)
    model = NTK(data.x_train, data.x_test, data.y_train, classes=10, depth=3)
    pred_test = model.predict(data.x_test, mode='test')
    test_loss = 1 / n_test * jnp.sum((pred_test - data.y_test)**2)
    if data.classes == 2:
        test_acc = 1 / n_test * jnp.sum(jnp.sign(pred_test) == data.y_test)
    else:
        test_acc = 1 / n_test * jnp.sum(jnp.argmax(pred_test, axis=1) == jnp.argmax(data.y_test, axis=1))
    loo_loss, loo_acc = model.leave_one_out()
    print('Test Loss is', test_loss)
    print('Leave-One-Out Error is', loo_loss)
    print('Test Accuracy is', test_acc)
    print('Leave-One-Out Accuracy is', loo_acc)
