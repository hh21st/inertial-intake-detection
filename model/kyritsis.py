"""Adaption of Kyitsis model for inertial data """

import tensorflow as tf

SCOPE = "kyritsis"

class Model(object):
    """Base class for Kyritsis model."""

    def __init__(self, params):
        self.num_classes = params.num_classes

    def __call__(self, inputs, is_training, scope=SCOPE):
        with tf.variable_scope(scope):

            inputs = tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=6,
                padding='same',
                activation=tf.nn.relu)(inputs)
            inputs = tf.keras.layers.MaxPool1D(
                pool_size=2)(inputs)
            inputs = tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=6,
                padding='same',
                activation=tf.nn.relu)(inputs)
            inputs = tf.keras.layers.MaxPool1D(
                pool_size=2)(inputs)
            inputs = tf.keras.layers.Dense(5)(inputs)
            inputs = tf.keras.layers.LSTM(
                    64, return_sequences=True)(inputs)
            inputs = tf.keras.layers.LSTM(
                    64, return_sequences=True)(inputs)
            inputs = tf.keras.layers.Dense(self.num_classes)(inputs)

            return inputs
