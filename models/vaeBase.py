import tensorflow as tf
import numpy as np
import os
from metrics import get_metrics


class VAEBase:
    
    
    def create_f_ss(self):

        self.ss_autoencoder = self.config['AutoencoderClass'](self.config)

        inp_shape = self.config['features_shape']
        ss_input = tf.keras.layers.Input(shape = inp_shape)
        ss_f_out = []
        
        for i in range(self.config['obs_l']):
            convu = self.ss_autoencoder.planar_encoder(ss_input[:, i, ...])
            ss_f_out.append(tf.expand_dims(convu, axis = 1))

        ss_f_conv_out = tf.concat(ss_f_out, axis = 1)
        ss_f_out = tf.keras.layers.Dense(self.config['f_dis_D'], activation = 'tanh')(ss_f_conv_out)

        self.f_ss = tf.keras.Model(ss_input, ss_f_out)
        self.ss_inp_shape = inp_shape
        
                
    def attach_data_transformer(self, transformer):
        
        self.transformer = transformer
    
    
    # make predictions by batches (in the case of use SS is not affordable make all the predictions at once)
    def decode_sequences(self, X, obj_pred):

        batch_size_eval = self.config['batch_size_vali']
        sequences = []

        # number of batches
        total_n = len(X['encoder'])
        total_b = total_n//batch_size_eval + int(total_n%batch_size_eval != 0)

        # iterate over each batch
        for i_batch in range(total_b):

            a = batch_size_eval * i_batch
            b = min(batch_size_eval * (1 + i_batch), total_n)

            # prepare inputs for decoder
            inputs_enc = [X['encoder'][a:b].astype('float32')]

            if self.config['use_features']:

                if (not 'ae_pre_enc' in self.config) or self.config['ae_pre_enc']:
                    inputs_enc.append(X['features'][a:b])
                else:
                    inputs_enc.append(self.config['get_semantic_batch_f'](self.config, X, [iii for iii in range(a, b)]))
            
                inputs_enc[-1] = inputs_enc[-1].astype('float32')    
                    
            # make prediction
            sequences.append(self.decode_batch_sequence(inputs_enc, obj_pred))

          # pack all the predictions

        pred = np.concatenate(sequences, axis = 0)
        del sequences

        assert len(pred) == len(X['encoder'])

        if 'transformer' in self.__dict__:
            pred = self.transformer.inverse_transform(pred.reshape(-1, self.config['D'])).reshape(pred.shape)
        else:
            assert self.config['transform_data'] == 0
            
        return pred


    def evaluate_model(self, input_data_x, input_data_y):

        pred = self.decode_sequences(input_data_x, self.config['outputs_validation'])
        assert len(pred) == len(input_data_y)

        y_gt_input = input_data_y

        _, ades, fdes = get_metrics(pred, y_gt_input)
        del pred
        return {"ADE" : np.mean(ades), "FDE" : np.mean(fdes)}


    def save_prediction_weights(self, r_path):

        self.encoder_inputs_model.save(os.path.join(r_path, 'enc_') + self.name_model + '.h5')
        self.decoder_model.save(os.path.join(r_path, 'dec_') + self.name_model + '.h5') 
        
        if 'ss_autoencoder' in self.__dict__:
            self.ss_autoencoder.planar_encoder.save(os.path.join(r_path, 'ss_enc_') + self.name_model + '.h5')
        
        print("MODEL {0} SAVED!".format(self.name_model))


    def load_prediction_weights(self, r_path): 

        dir_mod = os.path.join(r_path, 'enc_' + self.name_model + '.h5')
        self.encoder_inputs_model.load_weights(dir_mod)
        dir_mod = os.path.join(r_path, 'dec_' + self.name_model + '.h5')
        self.decoder_model.load_weights(dir_mod)
        
        if 'ss_autoencoder' in self.__dict__:
            dir_mod = os.path.join(r_path, 'ss_enc_') + self.name_model + '.h5'
            self.ss_autoencoder.planar_encoder.load_weights(dir_mod)

    
    
    def create_encoder(self, config):
    
        # displacements input
        self.X_inputs = tf.keras.layers.Input(shape = (None, config['D']))
        f_dis_X = tf.keras.Sequential([tf.keras.layers.Dense(config['f_dis_D'], activation = 'tanh',
                                            input_shape = (None, config['D'])),
                                      ])

        inputs = [self.X_inputs]
        
        # features input
        if config['use_features']:
            
            self.feat_inputs = tf.keras.layers.Input(shape = config['features_shape'])
            
            if (not 'ae_pre_enc' in config) or (config['ae_pre_enc']):
            
                f_feat = tf.keras.Sequential([tf.keras.layers.Dense(config['f_dis_D'], activation = 'tanh',
                                                   input_shape = config['features_shape'])])
            else:
                self.create_f_ss()
                f_feat = self.f_ss
                
            inputs.append(self.feat_inputs)
             
        # inputs encoder
        input_encoder = tf.keras.layers.LSTM(config['LSTM_D'], return_state = True)

        # prepare inputs
        if config['use_features']:
            encoder_inp = tf.concat([f_dis_X(self.X_inputs), f_feat(self.feat_inputs)], axis = -1)
        else:
            encoder_inp = f_dis_X(self.X_inputs)

        # encode inputs
        self.encoder_outputs, self.enc_s1, self.enc_s2 = input_encoder(encoder_inp)
        self.encoder_inputs_model = tf.keras.Model(inputs, [self.encoder_outputs, self.enc_s1, self.enc_s2]) 
    
    
    def create_decoder(self, config):
        
        # inputs
        z_inp = tf.keras.layers.Input(shape = (config['space_D'], )) # latent codes
        last_Xs1_inp = tf.keras.layers.Input(shape = (config['LSTM_D'], )) # lstm h. state 1
        last_Xs2_inp = tf.keras.layers.Input(shape = (config['LSTM_D'], )) # lstm h. state 2
        
        # this layers will be used to go to the space designated for the initial states of the LSTM decoder
        self.dense_decoder_s1 = tf.keras.layers.Dense(config['LSTM_D'])
        self.dense_decoder_s2 = tf.keras.layers.Dense(config['LSTM_D'])

        # decoder lstm
        self.decoder = tf.keras.layers.LSTM(config['LSTM_D'], return_sequences = True, return_state = True)    

        # layer to map the lstm output to IR^d, the space where de displacements belong
        self.dense_decode_D = tf.keras.layers.Dense(config['D'])
        
        # get the initial states for the decoder
        last_Xs1 = self.dense_decoder_s1(tf.concat([last_Xs1_inp, z_inp], 1))
        last_Xs2 = self.dense_decoder_s2(tf.concat([last_Xs2_inp, z_inp], 1))

        if config['use_teacherf']:

            # create a layer to embed the prediction to another space
            self.dense_pre_decoder = tf.keras.layers.Dense(config['f_dis_D'])
            
            traj = []

            last_X = self.X_inputs[:, -1] # last observation
            last_X = self.dense_pre_decoder(last_X)

            input_decoder = tf.expand_dims(last_X, 1)  

            # iterate to make the prediction

            for _ in range(config['pre_l']):

              # get output (not the prediction yet) and future states
              decoder_outputs, last_Xs1, last_Xs2 = self.decoder(input_decoder,
                                                                 initial_state = (last_Xs1, last_Xs2))

              # this is the prediction
              pred = self.dense_decode_D(decoder_outputs)
              traj.append(pred)

              # prepare next input
              input_decoder = self.dense_pre_decoder(pred)

            # create prediction sequence
            traj = tf.concat(traj, axis = 1)
            traj = tf.expand_dims(traj, axis = 1)

            self.decoder_model = tf.keras.Model([self.X_inputs, last_Xs1_inp, last_Xs2_inp, z_inp], traj)

        else:

            encoding_inp = tf.keras.layers.Input(shape = (config['LSTM_D'], )) # lstm output

            # repeat the output of the input encoder to do make the prediction
            input_decoder = tf.keras.layers.RepeatVector(config['pre_l'])(encoding_inp)

            decoder_outputs, _, _ = self.decoder(input_decoder,
                                                 initial_state = (last_Xs1, last_Xs2))

            # this is the prediction
            pred = self.dense_decode_D(decoder_outputs)
            traj = tf.expand_dims(pred, axis = 1)

            self.decoder_model = tf.keras.Model([encoding_inp, last_Xs1_inp, last_Xs2_inp, z_inp], traj)
