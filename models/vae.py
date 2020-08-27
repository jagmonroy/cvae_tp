import tensorflow as tf
import numpy as np
import gc
import os
from vaeBase import VAEBase
from losses import *
from metrics import get_metrics


class VAE(VAEBase):
    
    
    def __init__(self, config):
        
        self.config = config        
        
        self.name_model = 'vae_'
        if config['use_teacherf']: self.name_model += 'tf_'
        if config['use_gt_sampling']: self.name_model += 'gt_'
        if config['use_gt_sampling']: self.name_model += 'feat_'
        self.name_model += str(config['i_test']) + '_'
        self.name_model += str(config['run_id']) 

        self.create_encoder(config)
        self.create_decoder(config)    
    
        # in case of incorporate the target displacements to try to improve the properties of the latent space
        if config['use_gt_sampling']:

            self.Y_inputs = tf.keras.layers.Input(shape = (config['pre_l'], config['D']))
            self.f_dis_Y = tf.keras.Sequential([tf.keras.layers.Dense(config['f_dis_D'], activation = 'tanh',
                                                      input_shape = (None, config['D'])),
                                                # tf.keras.layers.BatchNormalization()
                                               ])

            self.gt_encoder = tf.keras.layers.LSTM(config['LSTM_D'], return_state = False)
            self.encoder_Y_out = self.gt_encoder(self.f_dis_Y(self.Y_inputs))
            pre_stats_input = tf.concat([self.encoder_outputs, self.encoder_Y_out], axis = 1)

        else:

            pre_stats_input = self.encoder_outputs

        # get stats to do the sampling
        pre_stats = tf.keras.layers.Dense(config['LSTM_D'], activation = 'tanh')(pre_stats_input)
        mu = tf.keras.layers.Dense(config['space_D'], activation = 'tanh')(pre_stats)
        logvar = tf.keras.layers.Dense(config['space_D'], activation = 'tanh')(pre_stats)
        std = tf.math.exp(0.5*logvar)

        # get a sample from the assumed distribution and use reparameterization trick
        eps = tf.random.normal(shape = tf.shape(mu))
        z = mu + eps*std

        # Needed to compute KL penalization
        mu = tf.expand_dims(mu, 0)
        logvar = tf.expand_dims(logvar, 0)
        self.stats = tf.concat([mu, logvar], 0)

        # training decoding

        decoder_X_s1 = self.dense_decoder_s1(tf.concat([self.enc_s1, z], 1))
        decoder_X_s2 = self.dense_decoder_s2(tf.concat([self.enc_s2, z], 1))
        initial_dec_states = (decoder_X_s1, decoder_X_s2)

        if config['use_teacherf']:

            inp_tf_dis = tf.keras.layers.Input(shape = (config['pre_l'], config['D']))
            input_decoder = self.dense_pre_decoder(inp_tf_dis)

        else:

            # repeat the output of the input encoder to do make the prediction
            input_decoder = tf.keras.layers.RepeatVector(config['pre_l'])(self.encoder_outputs)

        # decoder lstm
        decoder_output, _, _ = self.decoder(input_decoder, initial_state = initial_dec_states)
        outputs = self.dense_decode_D(decoder_output)
        outputs = tf.expand_dims(outputs, 1)

        # create model

        input_model = [self.X_inputs]

        if config['use_teacherf']:
            input_model.append(inp_tf_dis)

        if config['use_gt_sampling']:
            input_model.append(self.Y_inputs)

        if config['use_features']:
            input_model.append(self.feat_inputs)
            
        output_model = [outputs, self.stats]
        self.train_model = tf.keras.Model(input_model, output_model)
    
    
    def compile_model(self, config, optimizer):
    
        loss_list = [mse_loss, KL_loss]    

        self.train_model.compile(loss = loss_list,
                                 optimizer = optimizer,
                                 loss_weights = [1.0, config["kl_w"]])    

        
    def decode_batch_sequence(self, inputs_enc, preds_obj):
      
        enc_inp, s1, s2 = self.encoder_inputs_model(inputs_enc)

        preds = []

        if isinstance(preds_obj, int):
            it_n = preds_obj
        else:
            preds_noise = preds_obj
            it_n = len(preds_noise)

        # each iteration makes a prediction      
        for i in range(it_n):

            # latent codes
            if isinstance(preds_obj, int):
                noise_m = np.random.normal(size = (len(inputs_enc[0]), self.config['space_D'])).astype('float32')
            else:
                noise_m = preds_noise[i].astype('float32')

            if self.config['use_teacherf']:
                pred_m = self.decoder_model([inputs_enc[0], s1, s2, noise_m])
            else:
                pred_m = self.decoder_model([enc_inp, s1, s2, noise_m])

            preds.append(pred_m)
            del noise_m

        # pack all the predictions
        ans_pred = np.concatenate(preds, axis = 1)
        del preds
        return ans_pred  