import tensorflow as tf
import gc

# This callback is executed after each epoch and it is in charge of validation process.
# The best model according to mADE metric is saved and loaded in the end

class Callback(tf.keras.callbacks.Callback):

  def __init__(self, vae, x_input, y_input, config, save_model = 0):

    self.config = config
    self.save_model = save_model
    self.vae = vae
    
    self.x_input = x_input
    self.y_input = y_input
    
    self.best = None

    self.history = {}
    self.history['ADE'] = []
    self.history['FDE'] = []

    if self.save_model:
        print("SAVING MODEL:", vae.name_model)


  def on_epoch_end(self, epoch, logs = None):

      res_m = self.vae.evaluate_model(self.x_input, self.y_input)

      print('\n')
      for x in res_m:
        print(x, res_m[x])
      print('BEST:', self.best)
      print()

      self.history['ADE'].append(res_m['ADE'])
      self.history['FDE'].append(res_m['FDE'])
      gc.collect()

      if self.best is None or self.best > res_m['ADE']:
        self.best = res_m['ADE']
        
        if self.save_model:        
            self.vae.save_prediction_weights(self.config['r_path'])
