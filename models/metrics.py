import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from numba import jit
from tqdm.notebook import tqdm


def get_metrics(X, Y, selection = ('default', None), rela = True):

    assert len(X) == len(Y), 'len(X): {0}. len(Y): {1}.'.format(len(X), len(Y))

    if rela:
        X = np.cumsum(X, 2)
        Y = np.cumsum(Y, 1)
    else:
        X = np.array(X)
        Y = np.array(Y)
      
    if selection[0] == 'first':
        X = X[:, :selection[1], ...]
    elif selection[0] == 'cluster':
        
        ns = list(X.shape)
        ns[1] = selection[1]
        nX = np.zeros(ns)
        
        print("Total:", len(X))
        
        for i in tqdm(range(len(X))):
            nX[i] = select_trajs(selection[1], X[i])
            
        X = nX
        
        assert X.shape[1] == selection[1]
    
    dis = np.zeros_like(X)

    for i in range(X.shape[1]):
        dis[:, i, ...] = X[:, i, ...] - Y

    dis = np.linalg.norm(dis, axis = -1)

    ades = np.min(np.mean(dis, axis = -1), axis = -1)
    fdes = np.min(dis[:, :, -1], axis = -1)

    del dis

    return X, ades, fdes


@jit(nopython=True)
def select_trajs_2(n_select, M):

    IJ = np.argmax(M)
    i = int(IJ//len(M))
    j = int(IJ%len(M))
    
    max_t = [i, j]
    
    select = [max_t[0], max_t[1]]
    avail = np.zeros(len(M)) + 1
    avail[i] = 0
    avail[j] = 0

    for _ in range(n_select-2):

        max_d = -1
        max_i = 0

        for i in range(len(M)):

            if avail[i] == 0: continue

            min_d_s = np.inf

            for j in select:
                min_d_s = min(min_d_s, M[i, j])

            if max_d < min_d_s:
                max_d, max_i = min_d_s, i

        assert min_d_s > -1
        select.append(max_i)
        avail[max_i] = 0
     
    return avail
    

def select_trajs(n_select, trajs):
    
    num_trajs = trajs.shape[0]
    last = np.cumsum(trajs, -2)[:, -1]
    M = euclidean_distances(last, last)
    return trajs[select_trajs_2(n_select, M) == 0]
    
