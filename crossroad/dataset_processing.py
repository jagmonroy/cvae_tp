import tensorflow as tf

import numpy as np
import os
import tensorflow_addons as tfa
from sklearn.preprocessing import MinMaxScaler, normalize


# function taken from https://github.com/google/next-prediction (I removed and moved things)
def process_file(config, path_file_w):

  obs_len = config['obs_l']
  pred_len = config['pre_l']
  seq_len = config['tot_l']

  data = []
  num_person_in_start_frame = []
  seq_list = []
  seq_list_rel = []

  delim = ' '

  with open(path_file_w, "r") as traj_file:
    for line in traj_file:
      fidx, pid, x, y = line.split(delim)
      data.append([fidx, pid, x, y])
  data = np.array(data, dtype = "float32")
  
  # assuming the frameIdx is sorted in ASC
  frames = np.unique(data[:, 0]).tolist()  # all frame_idx
  frame_data = []  # people in frame
  for frame in frames:
    frame_data.append(data[frame == data[:, 0], :])

  idx = 0

  for frame in frames:
    
    cur_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis = 0)
    persons_in_cur_seq = np.unique(cur_seq_data[:, 1])
    num_person_in_cur_seq = len(persons_in_cur_seq)

    # the same shape in the next two np arrays
    cur_seq = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")        
    cur_seq_rel = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")

    # frameid for each seq timestep
    cur_seq_frame = np.zeros((num_person_in_cur_seq, seq_len), dtype="int32")

    count_person = 0

    for person_id in persons_in_cur_seq:
      
      cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]

      if len(cur_person_seq) != seq_len:
        continue

      # [seq_len,2]
      cur_person_seq = cur_person_seq[:, 2:]
      cur_person_seq_rel = np.zeros_like(cur_person_seq)

      # first frame is zeros x,y
      cur_person_seq_rel[1:, :] = cur_person_seq[1:, :] - cur_person_seq[:-1, :]
      cur_seq[count_person, :, :] = cur_person_seq
      cur_seq_rel[count_person, :, :] = cur_person_seq_rel

      # frame_idxs = frames[idx:idx+seq_len]
      # cur_seq_frame[count_person, :] = frame_idxs

      count_person += 1

    num_person_in_start_frame.append(count_person)
    # only count_person data is preserved
    seq_list.append(cur_seq[:count_person])
    seq_list_rel.append(cur_seq_rel[:count_person])

    idx += 1

  num_seq = len(seq_list)  # total number of frames across all videos (1 in this case)
  seq_list = np.concatenate(seq_list, axis=0)
  seq_list_rel = np.concatenate(seq_list_rel, axis=0)

  # we get the obs traj and pred_traj
  # [N*K, obs_len, 2]
  # [N*K, pred_len, 2]
  obs_traj = seq_list[:, :obs_len, :]
  pred_traj = seq_list[:, obs_len:, :]

  obs_traj_rel = seq_list_rel[:, :obs_len, :]
  pred_traj_rel = seq_list_rel[:, obs_len:, :]

  # the starting idx for each frame in the N*K list,
  # [num_frame, 2]
  cum_start_idx = [0] + np.cumsum(num_person_in_start_frame).tolist()
  seq_start_end = np.array([
    (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
  ], dtype="int")

  data = {
    "obs_traj": obs_traj,
    "obs_traj_rel": obs_traj_rel,
    "pred_traj": pred_traj,
    "pred_traj_rel": pred_traj_rel,
  }
  
  return data


def get_scene(folder, i):

    scene = os.path.join(folder, str(i) + '.npy')
    scene = np.load(scene)
    scene[scene == 1] = 0
    scene[scene == 2] = 1

    scene = (scene + 1) % 2

    ans = np.zeros(scene.shape + tuple([2]))

    n, m = scene.shape

    for i in range(n):
        for j in range(m):
            ans[i, j, scene[i, j]] = 1

    return ans


class LoadData:

    def __init__(self, config, verbose = True):

        semantic_folder = os.path.join(config['data_path'], 'segmentations')
        i_test = config['i_test']
        
        self.data = []

        trajs_folder = os.path.join(config['data_path'], 'trajs')
        files = os.listdir(trajs_folder)
        files.sort()
                
        for (i, file) in enumerate(files):
    
            f_path = os.path.join(trajs_folder, file)
            self.data.append(process_file(config, f_path))
        
        cut_x = ['obs_traj_rel']
        cut_y = ['pred_traj_rel']

        X_train, Y_train = self.create_dicts()
        X_test, Y_test = self.create_dicts()
        
        for (i, dat) in enumerate(self.data):
            for (xc, yc) in zip(cut_x, cut_y):

                x_decoder = np.zeros_like(dat[yc])
                x_decoder[:, 0, :] = np.array(dat[xc][:, -1, :])
                x_decoder[:, 1:, :] = np.array(dat[yc][:, 0:-1, :])

                x_decoder_traj = np.zeros_like(dat['pred_traj'])
                x_decoder_traj[:, 0, :] = np.array(dat['obs_traj'][:, -1, :])
                x_decoder_traj[:, 1:, :] = np.array(dat['pred_traj'][:, 0:-1, :])

                if i == i_test:

                    X_test['encoder'].append(np.array(dat[xc]))
                    X_test['decoder'].append(x_decoder)
                    X_test['scene_id'].append([i for _ in range(len(dat[xc]))])
                    X_test['semantics'][i] = get_scene(semantic_folder, i)
                    X_test['obs_traj'].append(np.array(self.data[i]['obs_traj']))
                    X_test['decoder_traj'].append(x_decoder_traj)

                    Y_test.append(np.array(dat['pred_traj_rel']))
                    
                else:
                    
                    X_train['encoder'].append(np.array(dat[xc]))
                    X_train['decoder'].append(x_decoder)
                    X_train['scene_id'].append([i for _ in range(len(dat[xc]))])
                    X_train['semantics'][i] = get_scene(semantic_folder, i)
                    X_train['obs_traj'].append(np.array(self.data[i]['obs_traj']))
                    X_train['decoder_traj'].append(x_decoder_traj)

                    Y_train.append(np.array(dat['pred_traj_rel']))
                    

        Y_train = self.concatenate_dics(X_train, Y_train)
        Y_test = self.concatenate_dics(X_test, Y_test)

        if verbose:
        
            self.print_shapes(X_train, Y_train, 'Train')
            self.print_shapes(X_test, Y_test, 'Test')

        if config['transform_data']:

            max_abs_value = config['max_abs_value']
            self.transformer = MinMaxScaler(feature_range = (-max_abs_value, max_abs_value))
            self.transformer = self.transformer.fit(X_train['encoder'].reshape([-1, X_train['encoder'].shape[-1]]))

            def transform_data(trans, data_sec):
                _, L, _ = data_sec.shape
                for i in range(L):
                    data_sec[:, i, :] = trans.transform(data_sec[:, i, :])

            transform_data(self.transformer, X_train['encoder'])
            transform_data(self.transformer, X_train['decoder'])
            transform_data(self.transformer, Y_train)

            transform_data(self.transformer, X_test['encoder'])
            
        self.X_train, self.Y_train = X_train, Y_train
        self.X_test, self.Y_test = X_test, Y_test


    def create_dicts(self):
        ans_X = {}
        ans_X['encoder'] = []
        ans_X['decoder'] = []
        ans_X['scene_id'] = []
        ans_X['semantics'] = {}
        ans_X['obs_traj'] = []
        ans_X['decoder_traj'] = []
        ans_Y = []
        return ans_X, ans_Y


    def concatenate_dics(self, X_dic, Y_dic):

        X_dic['encoder'] = np.concatenate(X_dic['encoder'], axis = 0)
        X_dic['decoder'] = np.concatenate(X_dic['decoder'], axis = 0)
        X_dic['scene_id'] = np.concatenate(X_dic['scene_id'], axis = 0)
        X_dic['obs_traj'] = np.concatenate(X_dic['obs_traj'], axis = 0)
        X_dic['decoder_traj'] = np.concatenate(X_dic['decoder_traj'], axis = 0)
        return np.concatenate(Y_dic, axis = 0)


    def print_shapes(self, X_dic, Y_dic, head):
        print('********************************************************')
        print(head)
        print('X_encoder:', X_dic['encoder'].shape)
        print('X_decoder:', X_dic['decoder'].shape)
        print('Y:', Y_dic.shape)
        print('********************************************************')


def get_semantic_batch_f(config, X, ids, key = 'obs_traj'):
        
    scenes = get_scenes(X, ids)
    return get_semantic_batch(scenes, X[key][ids], config['nLGrid'], config['fLGrid'], rot = False)

        
def get_semantic_batch(gPath, cPath, nLGrid, fLGrid, rot = False):
    
    if not rot:
        CS = (fLGrid, fLGrid)
    else:
        CS = (nLGrid, nLGrid)
        CSr = (fLGrid, fLGrid)
    
    a, b, _ = cPath.shape
    train_batch = np.zeros((a, b, *CS, gPath.shape[-1]), dtype = 'float32')
    tf_func_arg = tf.convert_to_tensor(gPath, dtype = 'float32')
    
    if rot:
        train_batch_r = np.zeros((a, b, *CSr, gPath.shape[-1]), dtype = 'float32')

    for i in range(cPath.shape[1]):
        c1 = cPath[:, i, ...] - nLGrid // 2
        c2 = (c1 + nLGrid) / gPath.shape[2]
        c1 = c1 / gPath.shape[2]

        boxes = [[a[1], a[0], b[1], b[0]] for (a, b) in zip(c1, c2)]
        box_ind = [i for i in range(len(boxes))]

        train_batch[:, i, ...] = tf.image.crop_and_resize(tf_func_arg, boxes, box_ind, CS)
        
        
        if rot:
            
            angs = np.random.uniform(low = 0.0, high = np.pi, size = len(train_batch[:, i, ...]))
            train_batch[:, i, ...] = tfa.image.rotate(tf.convert_to_tensor(train_batch[:, i, ...], dtype = 'float32'),
                                                      tf.convert_to_tensor(angs, dtype = 'float32'))
                
            train_batch_r[:, i, ...] = tf.image.resize(train_batch[:, i, ...], CSr)         
         
    if rot:
        ans = train_batch_r
    else:
        ans = train_batch

    f_s = ans.shape
    ans = ans.reshape((-1, 2))
    
    ind = np.sum(ans, axis = -1) == 0
    ans[ind] = [1, 0]
    ans = normalize(ans, norm = 'l1', axis = 1)
    ans = ans.reshape(f_s)        
    return ans


def get_scenes(X_dict, ids):
    n, m, c = X_dict['semantics'][X_dict['scene_id'][ids[0]]].shape

    ans_batch = np.zeros((len(ids), n, m, c))

    for i in range(len(ids)):
        ans_batch[i] = X_dict['semantics'][X_dict['scene_id'][ids[i]]]

    return ans_batch