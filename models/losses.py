import tensorflow as tf


def mse_loss(y, prediction):

    difs = []
    
    for i in range(prediction.shape[1]):
        difs.append(tf.expand_dims(tf.subtract(prediction[:, i, :, :], y), 1))
      
    dif = tf.concat(difs, 1)

    dif2 = tf.math.square(dif)
    
    xyloss = tf.math.reduce_sum(dif2, axis = 3)
    xyloss = tf.math.reduce_mean(xyloss, axis = 2)
    
    xyloss = tf.math.reduce_min(xyloss, axis = 1)      
    
    return tf.math.reduce_mean(xyloss)


def KL_loss(y, prediction):

    a = prediction[0]
    b = prediction[1]

    assert a.shape[0] == b.shape[0]

    # kl = -0.5 * K.sum(-K.exp(log_sigma) - K.square(mu) + 1.0 + log_sigma, axis=1)    
    return  - 0.5 * tf.reduce_mean(- tf.exp(b) - a**2 + 1 + b , axis = 1)
