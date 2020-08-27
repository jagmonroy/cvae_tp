import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from numba import jit


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


def get_by_id_and_frame(M):

    ids = list(set(M[:, 1]))
    traj_id = {}

    for id_p in ids:
        traj_id[id_p] = np.delete(M[M[:, 1] == id_p], 1, 1)

    frames = list(set(M[:, 0]))
    traj_fr = {}

    for frame in frames:
        traj_fr[frame] = np.delete(M[M[:, 0] == frame] , 0, 1)

    return traj_id, traj_fr


def get_trajs(config, trajs_id):

    trajs = {

        'id_p': [],
        'frames': [],
        'pos': []

    }

    for id_p in trajs_id:

        data = trajs_id[id_p]
        pos = len(data) - config['tot_l'] + 1

        if pos < 1: continue

        trajs['id_p'].extend([id_p for _ in range(pos)])
        trajs['frames'].extend([np.expand_dims(data[i:i+config['tot_l'], 0], axis = 0) for i in range(pos)])
        trajs['pos'].extend([np.expand_dims(data[i:i+config['tot_l'], 1:], axis = 0) for i in range(pos)])

    trajs['id_p'] = np.array(trajs['id_p'])
    trajs['frames'] = np.concatenate(trajs['frames'], axis = 0)
    trajs['pos'] = np.concatenate(trajs['pos'], axis = 0)

    return trajs

@jit(nopython = True)
def get_features(max_d, pos, frames, id_p, trajs_fr, len_fr, c_reduce):

    features = max_d + np.zeros((len(pos), len(pos[0]), 360))
    features_g = np.zeros((len(pos), len(pos[0]), c_reduce))
    window_s = 360//c_reduce

    for i in range(len(pos)):

        for j in range(len(pos[i])):

            frame = int(frames[i][j])
            all_p = trajs_fr[frame][:len_fr[frame]]
            id_c = id_p[i]

            point = all_p[all_p[:, 0] == id_c][0][1:]
            all_p = all_p[all_p[:, 0] != id_c][:, 1:]
            all_p = all_p - point
            all_p_c = all_p[:, 0] + 1j*all_p[:, 1]
            angles = np.angle(all_p_c, deg = True)
            angles += (angles<0)*360

            for k in range(len(angles)):

                ang = int(angles[k])
                features[i][j][ang] = min(features[i][j][ang], np.linalg.norm(all_p[k]))

            for k in range(c_reduce):
                features_g[i][j][k] = np.min(features[i][j][k*window_s:(k+1)*window_s])

    return features_g


def create_matrix(trajs_fr):

    a = int(np.max([x for x in trajs_fr]))+1
    b = int(np.max([len(trajs_fr[x]) for x in trajs_fr]))
    c = 3

    mat = np.zeros((a, b, c))
    len_m = np.zeros(len(mat))

    for i in trajs_fr:

        i = int(i)
        a = len(trajs_fr[i])
        len_m[i] = a
        mat[i, :a, :] = trajs_fr[i]

    for i in trajs_fr:
        i = int(i)
        assert len(trajs_fr[i]) == len_m[i], "buuuu with the lengths!"
        n = np.linalg.norm(trajs_fr[i] - mat[i, :int(len_m[i]), :])
        assert  n < 1e-1, "buuuu with the copy: {0}".format(n)

    return mat, len_m

def process_4_nn(config, trajs, trajs_fr, dataset):

    X = {
        'encoder': None,
        'decoder': None,
        'dataset': None,
        'pos': None,
        'frames': None,
        'id_p': None,
    }

    if config['use_features']:

        X.update({
            'obs_features': None,
            'pre_features': None,
        })

    Y = None

    if config['use_features']:

        trajs_fr_m, trajs_fr_l = create_matrix(trajs_fr)
        features = get_features(config['max_d'],
                                trajs['pos'], trajs['frames'], trajs['id_p'],
                                trajs_fr_m, trajs_fr_l, config['reduce'])
        X['obs_features'] = features[:, :config['obs_l']-1, :]
        X['pre_features'] = features[:, config['obs_l']-1:-1, :]

    X['pos'] = np.array(trajs['pos'])
    X['dataset'] = [dataset for _ in range(len(X['pos']))]
    X['frames'] = np.array(trajs['frames'])
    X['id_p'] = np.array(trajs['id_p'])

    obs = trajs['pos'][:, :config['obs_l']]
    pre = trajs['pos'][:, config['obs_l']:]

    X['encoder'] = obs[:, 1:] - obs[:, :-1]
    Y = pre - np.concatenate([obs[:, -1:], pre[:, :-1]], axis = 1)

    X['decoder'] = np.zeros_like(Y)
    X['decoder'][:, 0, :] = X['encoder'][:, -1, :]
    X['decoder'][:, 1:, :] = Y[:, :-1, :]

    return X, Y


def load_dataset(config, verbose):

    dataset_path = os.path.join(config['data_path'], config['i_test'])
    dictio = {}

    for phase in os.listdir(dataset_path):

        if verbose: print('   ', phase)

        dictio[phase] = {
            'X': {'encoder': [], 'decoder': [], 'dataset': [], 'pos': [], 'frames': [], 'id_p': []},
            'Y': [],
            'trajs_fr': {},
        }

        if config['use_features']:

            dictio[phase]['X'].update({
                'obs_features': [],
                'pre_features': [],
            })

        phase_path = os.path.join(dataset_path, phase)

        for file in os.listdir(phase_path):

            file_path = os.path.join(phase_path, file)
            if verbose: print('       ', file)

            mat = np.loadtxt(file_path)
            trajs_id, trajs_fr = get_by_id_and_frame(mat)

            trajs_ds = get_trajs(config, trajs_id)
            if verbose: print('       ', len(trajs_ds['pos']))

            X, Y = process_4_nn(config, trajs_ds, trajs_fr, file)

            dictio[phase]['X']['encoder'].append(X['encoder'])
            dictio[phase]['X']['decoder'].append(X['decoder'])
            dictio[phase]['X']['dataset'].append(X['dataset'])
            dictio[phase]['X']['pos'].append(X['pos'])
            dictio[phase]['X']['frames'].append(X['frames'])
            dictio[phase]['X']['id_p'].append(X['id_p'])

            if config['use_features']:

                dictio[phase]['X']['obs_features'].append(X['obs_features'])
                dictio[phase]['X']['pre_features'].append(X['pre_features'])

            dictio[phase]['Y'].append(Y)
            dictio[phase]['trajs_fr'][file] = trajs_fr

        if len(dictio[phase]) > 0:

            dictio[phase]['Y'] = np.concatenate(dictio[phase]['Y'], axis = 0)
            dictio[phase]['X']['encoder'] = np.concatenate(dictio[phase]['X']['encoder'], axis = 0)
            dictio[phase]['X']['decoder'] = np.concatenate(dictio[phase]['X']['decoder'], axis = 0)
            dictio[phase]['X']['dataset'] = np.concatenate(dictio[phase]['X']['dataset'], axis = 0)
            dictio[phase]['X']['pos'] = np.concatenate(dictio[phase]['X']['pos'], axis = 0)
            dictio[phase]['X']['frames'] = np.concatenate(dictio[phase]['X']['frames'], axis = 0)
            dictio[phase]['X']['id_p'] = np.concatenate(dictio[phase]['X']['id_p'], axis = 0)

            if config['use_features']:

                dictio[phase]['X']['features'] = np.concatenate(dictio[phase]['X']['obs_features'], axis = 0)
                dictio[phase]['X']['pre_features'] = np.concatenate(dictio[phase]['X']['pre_features'], axis = 0)

                dictio[phase]['X']['features'] /= config['max_d']
                dictio[phase]['X']['pre_features'] /= config['max_d']
                
                        
        if verbose: print('   ', len(dictio[phase]['Y']))

    return dictio
    
    
class LoadData:
    
    def __init__(self, config, verbose = 1):
        
        self.data = load_dataset(config, verbose = verbose)
        
        self.X_train = self.data['train']['X']
        self.Y_train = self.data['train']['Y']

        self.X_test = self.data['test']['X']
        self.Y_test = self.data['test']['Y']

        if config['formal_training']:
            
            self.X_vali = self.data['val']['X']
            self.Y_vali = self.data['val']['Y']

    