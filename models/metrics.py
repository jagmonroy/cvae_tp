import numpy as np


def get_metrics(X, Y, rela = True):

    assert len(X) == len(Y)

    if rela:
      X = np.cumsum(X, 2)
      Y = np.cumsum(Y, 1)
    else:
      X = np.array(X)
      Y = np.array(Y)

    dis = np.zeros_like(X)

    for i in range(X.shape[1]):
      dis[:, i, ...] = X[:, i, ...] - Y

    dis = np.linalg.norm(dis, axis = -1)

    ades = np.min(np.mean(dis, axis = -1), axis = -1)
    fdes = np.min(dis[:, :, -1], axis = -1)

    del X, Y, dis
    return ades, fdes
