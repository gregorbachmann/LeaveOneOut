import numpy as np
import jax.numpy as jnp


def pseudo_divide(a, b, rank):
    n = b.shape[0]
    b = np.concatenate([np.zeros_like(b[rank:]), b[n-rank:]])
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


def accuracy(preds, targs):
    num_samples, num_out = preds.shape
    if num_out == 1:
        acc = jnp.mean(jnp.sign(preds) == targs)
    else:
        acc = 1 / num_samples * jnp.sum(jnp.argmax(preds, axis=1) == jnp.argmax(targs, axis=1), axis=0)

    return acc


def mse_loss(preds, targs):
    num_samples, num_out = preds.shape
    if num_out == 1:
        loss = jnp.mean((preds - targs)**2)
    else:
        loss = 1 / num_samples * jnp.sum((preds - targs)**2)

    return loss
