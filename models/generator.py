import tensorflow as tf
import numpy as np
import gc


class Generator(tf.keras.utils.Sequence):
    
    def __init__(self, Xd, Y, config):

        self.config = config

        self.X = Xd
        self.Y = Y
        self.list_IDs = [i for i in range(len(self.X['encoder']))]
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.config['batch_size']))

    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.config['batch_size']:(index+1)*self.config['batch_size']]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        batch = self.__data_generation(list_IDs_temp)

        return batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.config['shuffle'] == True:
            np.random.shuffle(self.indexes)
        gc.collect()

    def __data_generation(self, list_IDs_temp):

        X = self.X['encoder'][list_IDs_temp]
        y = self.Y[list_IDs_temp]  
          
        inputs = [X]
        
        if self.config['model'] == 'sdvae':
            inputs.append(y)
        
        if self.config['use_teacherf']:
        
            X_tf = self.X['decoder'][list_IDs_temp]
            inputs.append(X_tf)
        
        if self.config['use_gt_sampling']:
            inputs.append(np.array(y))

        if self.config['use_features']:
            
            if (not 'ae_pre_enc' in self.config) or self.config['ae_pre_enc']:
                inputs.append(self.X['features'][list_IDs_temp])
            else:
                inputs.append(self.config['get_semantic_batch_f'](self.config, self.X, list_IDs_temp))
            
        if self.config['model'] == 'vae':
            return inputs, [y, np.array([0])]
        else:
            return inputs
        
        
    def data_generation(self, list_IDs_temp):
        return self.__data_generation(list_IDs_temp)
