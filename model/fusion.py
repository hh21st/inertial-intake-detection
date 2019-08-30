"""Fusion model for inertial data"""

import numpy as np
import tensorflow as tf
import cnn
import gru
import lstm
import blstm

def fuse(params, inputs1, inputs2, is_training):
    
    cnn1 = cnn.Model(params) 
    seq_pool1, inputs1 = cnn1(inputs1, '_accel', is_training)

    cnn2 = cnn.Model(params) 
    seq_pool2, inputs2 = cnn2(inputs2, '_gyro', is_training)

    assert seq_pool1 == seq_pool2, 'seq_pool for accel is {} whereas seq_pool for gyro is {}'.format(seq_pool1, seq_pool2)

    if params.model == 'cnn_lstm':
        rnn1 = lstm.Model(params)
        rnn2 = lstm.Model(params)
    elif params.model == 'cnn_gru':
        rnn1 = gru.Model(params)
        rnn2 = gru.Model(params)
    elif params.model == 'cnn_blstm':
        rnn1 = blstm.Model(params)
        rnn2 = blstm.Model(params)

    inputs1 = rnn1(inputs1, is_training)
    inputs1 = tf.nn.softmax(inputs1, name='softmax_tensor')
    
    inputs2 = rnn2(inputs2, is_training)
    inputs2 = tf.nn.softmax(inputs2, name='softmax_tensor')
    
    return seq_pool1, inputs1, inputs2


def lr(params, inputs, is_training):
    
    inputs_left = inputs[:,:,0:6]
    inputs_dom_hand_flag = inputs[:,:,12:13]
    inputs_left = tf.concat([inputs_left, inputs_dom_hand_flag], axis = 2)
    inputs_right = inputs[:,:,6:13]

    return fuse(params, inputs_left, inputs_right, is_training)

def lr1(params, inputs, is_training):
    seq_pool, inputs_left, inputs_right = lr(params, inputs, is_training)
    inputs = inputs_left + inputs_right

    return seq_pool, inputs

def ag(params, inputs, is_training):
    
    inputs_accel_left = inputs[:,:,0:3] # 0,1,2
    inputs_gyro_left = inputs[:,:,3:6] # 3,4,5
    inputs_accel_right = inputs[:,:,6:9] # 6,7,8 
    inputs_gyro_right_and_dom = inputs[:,:,9:13] # 9,10,11,12
    inputs_dom_hand_flag = inputs[:,:,12:13]

    inputs_accel = tf.concat([inputs_accel_left, inputs_accel_right, inputs_dom_hand_flag], axis = 2)
    inputs_gyro = tf.concat([inputs_gyro_left, inputs_gyro_right_and_dom], axis = 2)

    return fuse(params, inputs_accel, inputs_gyro, is_training)

def ag1(params, inputs, is_training):
    seq_pool, inputs_accel, inputs_gyro = ag(params, inputs, is_training)
    inputs = inputs_accel + inputs_gyro

    return seq_pool, inputs

def ag2(params, inputs, is_training):
    seq_pool, inputs_accel, inputs_gyro = ag(params, inputs, is_training)
    inputs = inputs_accel + inputs_gyro + inputs_gyro

    return seq_pool, inputs
    
class Model(object):
    """Base class for Modality CNN BLSTM model."""

    def __init__(self, params):
        self.params = params
    def __call__(self, inputs, is_training):
        if self.params.fusion == 'accel_gyro':
            if self.params.f_mode == 'ag1':
                return ag1(self.params, inputs, is_training)
            if self.params.f_mode == 'ag2':
                return ag2(self.params, inputs, is_training)
        elif self.params.fusion == 'left_right':
            if self.params.f_mode == 'lr1': 
                return lr1(self.params, inputs, is_training)