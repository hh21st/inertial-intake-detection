"""CNN LSTM for inertial data"""

import tensorflow as tf

SCOPE = "cnn_lstm"

class Model(object):
    """Base class for 7 layer CNN plus 2 layer LSTM model."""

    def __init__(self, params):
        self.num_classes = params.num_classes
    def __call__(self, inputs, is_training, scope=SCOPE):
        with tf.variable_scope(scope):
            seq_pool=1
            inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
            inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
            seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
            inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
            inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
            seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
            inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
            inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
            seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
            inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
            inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
            seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8
            #inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
            #inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
            #seq_pool=seq_pool*2 # seq_pool=32 => seq_length= 128 /32 = 4
            #inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
            #inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
            #seq_pool=seq_pool*2 # seq_pool=64 => seq_length= 128 /64 = 2
            #inputs = tf.keras.layers.Conv1D(filters=512, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
            #inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
            #seq_pool=seq_pool*2 # seq_pool=128 => seq_length= 128 /128 = 1

            inputs = tf.keras.layers.Dense(8)(inputs)

            inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
            inputs = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)

            inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
            return seq_pool, inputs
