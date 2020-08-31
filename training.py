import sys
import os

assert len(sys.argv) == 2, 'Only one argument is required.'
assert os.path.exists(sys.argv[1]), 'Argument does not correspond to an existing file.'
assert sys.argv[1][-4:] == 'yaml', 'File does not have a yaml extension.'

from training_utils import *
config = load_config_file(sys.argv[1], 1)

sys.path.append('models')
if 'crossroad' in config['data_path']:
    sys.path.append('crossroad')
elif 'ETH-UCY' in config['data_path']:
    sys.path.append('ETH-UCY')
else:
    assert 1==0, 'No ETH-UCY or crossroad'

import tensorflow as tf
import numpy as np
import itertools
import yaml
from vae import *
from sdvae import *
from generator import *
from callback import *
from autoencoder_patches import *
from dataset_processing import *
from metrics import *

if __name__ == '__main__':

    for args in itertools.product(config['test_list'], config['id_list']):

        tf.keras.backend.clear_session()

        change_args = {

            'i_test': args[0],
            'run_id': args[1],

        }

        config.update(change_args)
        data = LoadData(config)

        if config['use_features'] and 'ae_pre_enc' in config:

            config['get_semantic_batch_f'] = get_semantic_batch_f
            config['AutoencoderClass'] = Autoencoder

        if config['use_features'] and ('ae_pre_enc' in config and config['ae_pre_enc']):

            save_dir = os.path.join(config['data_path'], 'pre_encodings')        
            os.makedirs(save_dir, exist_ok = True)
            load_file = os.path.join(save_dir, str(config['i_test'])) + '.npy'

            if os.path.exists(load_file) == False:

                print('Training autoencoder')

                autoencoder = Autoencoder(config)
                h, _, _ = autoencoder.train(config, data)
                h = h.history

                ae_name = os.path.join(save_dir, 'enc_' + str(config['i_test']) + '.h5')
                autoencoder.planar_encoder.save(ae_name)
                ae_name = os.path.join(save_dir, 'dec_' + str(config['i_test']) + '.h5')
                autoencoder.tran_dec.save(ae_name)

                ae_h = os.path.join(save_dir, 'h_' + str(config['i_test']))
                np.save(ae_h, h)

                print('Doing pre-encoding')

                data.X_train['features'] = get_features(autoencoder, data.X_train, 'obs_traj', config)
                assert len(data.X_train['features']) == len(data.X_train['encoder']) 

                data.X_train['features_dec'] = get_features(autoencoder, data.X_train, 'decoder_traj', config)
                assert len(data.X_train['features_dec']) == len(data.X_train['encoder'])

                data.X_test['features'] = get_features(autoencoder, data.X_test, 'obs_traj', config)
                assert len(data.X_test['features']) == len(data.X_test['encoder'])

                np.save(load_file, data)

                tf.keras.backend.clear_session()

            data = np.load(load_file, allow_pickle = True)
            data = data.item()

            print('Pre-encoding loaded!')

        if config['use_features']:
            if 'reduce' in config:
                config['features_shape'] = (None, data.X_train['features'].shape[-1])
            elif 'num_filters' in config:
                if config['ae_pre_enc']:
                    config['features_shape'] = (None, data.X_train['features'].shape[-1])
                else:
                    config['features_shape'] =  (None, config['fLGrid'], config['fLGrid'], config['n_classes'])

        if config['model'] == 'vae':
            vae_m = VAE(config)
        else:
            vae_m = SDVAE(config)

        if 'transform_data' in config and config['transform_data']:
            vae_m.attach_data_transformer(data.transformer)

        train_gen = Generator(data.X_train, data.Y_train, config)
        callbacks = get_callbacks(config, vae_m, data)
        optimizer = get_optimizer(config)
        vae_m.compile_model(config, optimizer)

        r = vae_m.train_model.fit(train_gen,
                                  epochs = config['epochs'],
                                  callbacks = callbacks,
                                  use_multiprocessing = False,
                                  workers = 1,
                                  verbose = 1
                                  )

        save_history(config['r_path'], r, callbacks, vae_m.name_model)
        vae_m.load_prediction_weights(config['r_path'])
        pred_t = vae_m.decode_sequences(data.X_test, config['outputs_final_test'])
        f_name = os.path.join(config['r_path'], vae_m.name_model) + '_preds'
        np.save(f_name, pred_t)
        print('Saved in', f_name)

        _, ades, fdes = get_metrics(pred_t, data.Y_test)
        print('ade', np.mean(ades))
        print('fde', np.mean(fdes))
        print()

        f_name = os.path.join(config['r_path'], vae_m.name_model) + '_config.yaml'
        f = open(f_name, "w")
        yaml.dump(config, f)
        f.close()
