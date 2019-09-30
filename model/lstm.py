"""LSTM for inertial data"""

import tensorflow as tf

def cl1(self, inputs, is_training):
    
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    
    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl2(self, inputs, is_training):
    
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    
    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl3(self, inputs, is_training):
    
    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    
    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl3_0(self, inputs, is_training):
    
    inputs = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    
    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl3_0_1(self, inputs, is_training):
    
    inputs = tf.keras.layers.LSTM(1024, return_sequences=True)(inputs)
    
    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl3_0_2(self, inputs, is_training):
    
    inputs = tf.keras.layers.LSTM(1024, return_sequences=True)(inputs)
    
    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl3_1(self, inputs, is_training):
    
    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    
    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl3_1_0(self, inputs, is_training):
    
    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    
    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl3_1_1(self, inputs, is_training):
    
    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl3_1_1_0(self, inputs, is_training):

    inputs = tf.keras.layers.LSTM(512, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl3_2(self, inputs, is_training):

    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl3_2_1(self, inputs, is_training):

    inputs = tf.keras.layers.Dense(2)(inputs)

    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl4(self, inputs, is_training):

    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl4_1(self, inputs, is_training):

    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cl5(self, inputs, is_training):

    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def d13_nd(self, inputs, is_training):
    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def d13_nd_2(self, inputs, is_training):
    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs


class Model(object):
    """Base class for LSTM model."""

    def __init__(self, params):
        self.params = params
        self.num_classes = params.num_classes
        self.sub_mode = params.sub_mode
    def __call__(self, inputs, is_training):
        var_scope = 'lstm'
        with tf.variable_scope(var_scope):
            if self.sub_mode == 'cl1':
                return cl1(self, inputs, is_training)
            elif self.sub_mode == 'cl2':
                return cl2(self, inputs, is_training)
            elif self.sub_mode == 'cl3':
                return cl3(self, inputs, is_training)
            elif self.sub_mode == 'cl3_0':
                return cl3_0(self, inputs, is_training)
            elif self.sub_mode == 'cl3_0_1':
                return cl3_0_1(self, inputs, is_training)
            elif self.sub_mode == 'cl3_0_2':
                return cl3_0_2(self, inputs, is_training)
            elif self.sub_mode == 'cl3_1':
                return cl3_1(self, inputs, is_training)
            elif self.sub_mode == 'cl3_1_nd':
                return cl3_1(self, inputs, is_training)
            elif self.sub_mode == 'cl3_1_0':
                return cl3_1_0(self, inputs, is_training)
            elif self.sub_mode == 'cl3_1_1':
                return cl3_1_1(self, inputs, is_training)
            elif self.sub_mode == 'cl3_1_1_0':
                return cl3_1_1_0(self, inputs, is_training)
            elif self.sub_mode == 'cl3_2':
                return cl3_2(self, inputs, is_training)
            elif self.sub_mode == 'cl3_2_1':
                return cl3_2_1(self, inputs, is_training)
            elif self.sub_mode == 'cl4':
                return cl4(self, inputs, is_training)
            elif self.sub_mode == 'cl4_1':
                return cl4_1(self, inputs, is_training)
            elif self.sub_mode == 'cl5':
                return cl5(self, inputs, is_training)
            elif self.sub_mode == 'd13_nd':
                return d13_nd(self, inputs, is_training)
            elif self.sub_mode == 'd13_nd_2':
                return d13_nd_2(self, inputs, is_training)
            else:
                raise RuntimeError('sub mode {0} is not implemented'.format(self.sub_mode))

