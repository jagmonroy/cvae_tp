
import tensorflow as tf
import numpy as np
import os
from vaeBase import VAEBase
from losses import mse_loss


class SDVAE(VAEBase):

    def __init__(self, config):
        
        self.config = config
            
        self.name_model = 'sdvae_'
        if config['use_teacherf']: self.name_model += 'tf_'
        if config['use_gt_sampling']: self.name_model += 'gt_'
        if config['use_gt_sampling']: self.name_model += 'feat_'
        self.name_model += str(config['i_test']) + '_'
        self.name_model += str(config['run_id']) 


        self.create_encoder(config)
        self.create_decoder(config)
        
        # training decoding
        
        if config['use_teacherf']:

            inp_tf_dis = tf.keras.layers.Input(shape = (config['pre_l'], config['D']))
            input_decoder = self.dense_pre_decoder(inp_tf_dis)
        
        else:
            
            # repeat the output of the input encoder to do make the prediction
            input_decoder = tf.keras.layers.RepeatVector(config['pre_l'])(self.encoder_outputs)
        
        outputs = []

        for i in range(config['space_D']):

            z = tf.one_hot([i for _ in range(config['batch_size'])], config['space_D'])
            decoder_X_s1 = self.dense_decoder_s1(tf.concat([self.enc_s1, z], 1))
            decoder_X_s2 = self.dense_decoder_s2(tf.concat([self.enc_s2, z], 1))
            initial_dec_states = (decoder_X_s1, decoder_X_s2)

            decoder_output, _, _ = self.decoder(input_decoder, initial_state = initial_dec_states)
            outputs_i = self.dense_decode_D(decoder_output)

            # prepare the output in case of multiple outcomes
            outputs.append(tf.expand_dims(outputs_i, 1))

        self.pred = tf.concat(outputs, 1, name = 'all_outputs')

        # loss
        
        self.Y_inputs = tf.keras.layers.Input(shape = (config['pre_l'], config['D']))
        input_model = [self.X_inputs, self.Y_inputs]

        if config['use_teacherf']:
            input_model.append(inp_tf_dis)

        if config['use_features']:
            input_model.append(self.feat_inputs)

        self.train_model = tf.keras.Model(input_model, self.pred)
        self.train_model.add_loss(mse_loss(self.Y_inputs, self.pred))
                
        
    def compile_model(self, config, optimizer):
    
        self.train_model.compile(optimizer = optimizer)
        
        
    def decode_batch_sequence(self, inputs_enc, num_preds):
      
        assert isinstance(num_preds, int), 'SDVAE decode sequence function have to receive an integer.'

        enc_inp, s1, s2 = self.encoder_inputs_model(inputs_enc)

        preds = []

        # each iteration makes a prediction      
        for i in range(num_preds):

            # latent codes
            noise_m = np.zeros((len(inputs_enc[0]), self.config['space_D']))
            noise_m[:, i] = 1
            noise_m = noise_m.astype('float32')

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