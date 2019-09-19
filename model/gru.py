"""GRU for inertial data"""

import tensorflow as tf

def cg1(self, inputs, is_training):

    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs


def cg2(self, inputs, is_training):

    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cg3(self, inputs, is_training):

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cg3_1(self, inputs, is_training):

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cg3_1_1(self, inputs, is_training):

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cg3_2(self, inputs, is_training):

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cg3_2_1(self, inputs, is_training):

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cg4(self, inputs, is_training):

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cg4_1(self, inputs, is_training):

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs

def cg5(self, inputs, is_training):

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return inputs


class Model(object):
    """Base class for GRU model."""

    def __init__(self, params):
        self.params = params
        self.num_classes = params.num_classes
        self.sub_mode = params.sub_mode
    def __call__(self, inputs, is_training):
        var_scope = 'gru'
        with tf.variable_scope(var_scope):
            if self.sub_mode == 'cg1':
                return cg1(self, inputs, is_training)
            elif self.sub_mode == 'cg2':
                return cg2(self, inputs, is_training)
            elif self.sub_mode == 'cg3':
                return cg3(self, inputs, is_training)
            elif self.sub_mode == 'cg3_1':
                return cg3_1(self, inputs, is_training)
            elif self.sub_mode == 'cg3_1_1':
                return cg3_1_1(self, inputs, is_training)
            elif self.sub_mode == 'cg3_2':
                return cg3_2(self, inputs, is_training)
            elif self.sub_mode == 'cg3_2_1':
                return cg3_2_1(self, inputs, is_training)
            elif self.sub_mode == 'cg4':
                return cg4(self, inputs, is_training)
            elif self.sub_mode == 'cg4_1':
                return cg4_1(self, inputs, is_training)
            elif self.sub_mode == 'cg5':
                return cg5(self, inputs, is_training)
            else:
                raise RuntimeError('sub mode {0} is not implemented'.format(self.sub_mode))

