import tensorflow as tf
import numpy as np
from tqdm.notebook import tqdm


def get_features(autoencoder, X, key, config):

    get_semantic_batch_f = config['get_semantic_batch_f']
    planar_time = tf.keras.layers.TimeDistributed(autoencoder.planar_encoder)

    ans = []
    n = len(X['encoder'])
    n_enc = n//config['ae_bs'] + (n%config['ae_bs']!=0)

    for i in tqdm(range(n_enc)):

        a = i*config['ae_bs']
        b = min((i+1)*config['ae_bs'], n)

        ids = [j for j in range(a, b)]
        batch = get_semantic_batch_f(config, X, ids, key = key)
        feats = planar_time(batch)
        ans.append(feats)

    ans = np.concatenate(ans, axis = 0)
    return ans


class PatchesDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, X, config, augmentation = False):

        self.augmentation = augmentation
        self.batch_size = config['ae_bs']
        self.shuffle = config["shuffle"]
        self.polar_c = config["polar_c"]
        self.get_semantic_batch_f = config['get_semantic_batch_f']
        
        self.config = config
        
        self.X = X
        self.list_IDs = np.array([i for i in range(len(self.X['obs_traj']))])

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
                
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = self.list_IDs[indexes]
        X = self.__data_generation(list_IDs_temp)
        return X, X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        X = self.get_semantic_batch_f(self.config, self.X, list_IDs_temp)
        X = X.reshape(tuple([-1]) + X.shape[-3:])

        if self.polar_c:

            for i in range(X.shape[0]):
                img = X[i].astype(np.float32)
                value = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
                p_ima = cv2.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)
                X[i] = np.array(p_ima)

        return X

    def data_generation(self, list_IDs_temp):
        return self.__data_generation(list_IDs_temp)
    
    
class Autoencoder:

    def __init__(self, config):

        num_filters = config['num_filters']
        k_size = tuple(config['filter_size'])
        self.config = config
        
        self.conv_enc = [tf.keras.layers.BatchNormalization()]
        
        for _ in range(config['n_layers']):
        
            self.conv_enc.extend([
                tf.keras.layers.Conv2D(num_filters, k_size, activation = 'tanh', padding = 'same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
            ])
        
        self.tran_dec = []
        
        for _ in range(config['n_layers']):
        
            self.tran_dec.extend([
                tf.keras.layers.Conv2DTranspose(num_filters,
                                                k_size,
                                                strides = 2,
                                                activation = 'tanh',
                                                padding = 'same'),
            ])
             
        self.tran_dec.append(
             tf.keras.layers.Conv2D(config['n_classes'], k_size, activation = 'softmax', padding = 'same')
        )
        
        # ************************************* Input ****************************************
        
        input_conv_shape = (config['fLGrid'], config['fLGrid'], config['n_classes'])
        inp = tf.keras.layers.Input(shape = input_conv_shape)

        # ************************************* Encoder **************************************
        
        conv_enc = tf.keras.models.Sequential(self.conv_enc)
        
        conv_enc_inp = conv_enc(inp)
        conv_enc_inp_f = tf.keras.layers.Flatten()(conv_enc_inp)
            
        self.planar_encoder = tf.keras.models.Model(inp, conv_enc_inp_f)
                
        # ************************************* Decoder **************************************   
  
        self.tran_dec = tf.keras.models.Sequential(self.tran_dec)
        inp_dec = self.tran_dec(conv_enc_inp)

        # ************************************* Train model **********************************   
        
        self.ae = tf.keras.models.Model(inp, inp_dec)

        # ************************************* Loss function ********************************   
        
        rec_loss_f = tf.keras.losses.KLDivergence()
        rec_loss = rec_loss_f(inp, inp_dec)
        
        self.ae.add_loss(rec_loss)


    def train(self, config, data):
        
        # ************************************* Training *************************************   
        
        optimizer = tf.keras.optimizers.Adam(lr = config['ae_lr'])
        self.ae.compile(optimizer = optimizer)
        
        train_gen = PatchesDataGenerator(data.X_train, config, augmentation = config['ae_augmentation'])
        vali_gen = PatchesDataGenerator(data.X_test, config, augmentation = False)

        h = self.ae.fit(train_gen, validation_data = vali_gen, epochs = config['ae_epochs'],
                        use_multiprocessing = False,
                        workers = 1,
                       )

        return h, train_gen, vali_gen

