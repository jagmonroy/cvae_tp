import tensorflow as tf
import numpy as np
import yaml
import os
import sys

sys.path.append('models')

from generator import Generator
from callback import Callback


def load_config_file(file_path, create_folder_results):

    stream = open(file_path, 'r')
    config = yaml.safe_load(stream)
    stream.close()
    
    # direction where the results are saved
    config["r_path"] = os.path.join(config['results_path'], config['prefix'] + "_")
    config["r_path"] += str(config['use_teacherf']) + '_'
    config["r_path"] += str(config['use_gt_sampling']) + '_'
    config["r_path"] += str(config['use_features'])

    # model name
    if create_folder_results:
        os.makedirs(config['r_path'], exist_ok = True)
        
    return config


def get_callbacks(config, vae_m, data):
        
    if config['formal_training']:
        
        callback_1 = Callback(vae_m,
                                  data.X_vali, data.Y_vali,
                                  config,
                                  save_model = True)

        callback_2 = Callback(vae_m,
                                  data.X_test, data.Y_test,
                                  config,
                                  save_model = False)

        return [callback_1, callback_2]

    callback = Callback(vae_m,
                        data.X_test, data.Y_test,
                        config,
                        save_model = True)

    return [callback]


def get_optimizer(config):
    
    if config['sgd']:
        optimizer = tf.optimizers.SGD(learning_rate = config['lr'],
                                      momentum = config['momentum'], 
                                      decay = config['decay'])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate = config['lr'])
    
    return optimizer


def save_history(path, r, callbacks, name_model):
    
    if len(callbacks) == 2:
    
        r.history['vali_metrics'] = callbacks[0].history
        r.history['test_metrics'] = callbacks[1].history

    else:
        r.history['test_metrics'] = callbacks[0].history

    np.save(os.path.join(path, name_model + '_history'), r.history)


