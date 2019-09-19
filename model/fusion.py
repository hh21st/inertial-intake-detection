"""Fusion model for inertial data"""

import numpy as np
import tensorflow as tf
import cnn
import gru
import lstm
import blstm

def fuse_late(params, inputs1, inputs2, is_training):
    
    cnn1 = cnn.Model(params) 
    seq_pool1, inputs1 = cnn1(inputs1, '_inputs1', is_training)

    cnn2 = cnn.Model(params) 
    seq_pool2, inputs2 = cnn2(inputs2, '_inputs2', is_training)

    assert seq_pool1 == seq_pool2, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 2 is {}'.format(seq_pool1, seq_pool2)

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

def fuse_early(params, inputs1, inputs2, is_training):
    
    cnn1 = cnn.Model(params) 
    seq_pool1, inputs1 = cnn1(inputs1, '_inputs1', is_training)

    cnn2 = cnn.Model(params) 
    seq_pool2, inputs2 = cnn2(inputs2, '_inputs2', is_training)

    assert seq_pool1 == seq_pool2, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 2 is {}'.format(seq_pool1, seq_pool2)

    inputs = tf.concat([inputs1, inputs2], axis = 2)

    if params.model == 'cnn_lstm':
        rnn = lstm.Model(params)
    elif params.model == 'cnn_gru':
        rnn = gru.Model(params)
    elif params.model == 'cnn_blstm':
        rnn = blstm.Model(params)

    inputs = rnn(inputs, is_training)
    return seq_pool1, inputs


def dnd(params, inputs, is_training):
    
    inputs_dom = inputs[:,:,0:6]
    inputs_ndom = inputs[:,:,6:12]

    if params.f_strategy == 'late':
        return fuse_late(params, inputs_dom, inputs_ndom, is_training)
    elif params.f_strategy == 'early':
        return fuse_early(params, inputs_dom, inputs_ndom, is_training)
    else:
        raise RuntimeError('{0} is not supported in fusion.py'.format(params.f_strategy))

def dnd1(params, inputs, is_training):
    seq_pool, inputs_left, inputs_right = dnd(params, inputs, is_training)
    inputs = inputs_left + inputs_right

    return seq_pool, inputs

def ag(params, inputs, is_training):
    
    inputs_accel_dom = inputs[:,:,0:3] # 0,1,2
    inputs_gyro_dom = inputs[:,:,3:6] # 3,4,5
    inputs_accel_ndom = inputs[:,:,6:9] # 6,7,8 
    inputs_gyro_ndom = inputs[:,:,9:12] # 9,10,11

    inputs_accel = tf.concat([inputs_accel_dom, inputs_accel_ndom], axis = 2)
    inputs_gyro = tf.concat([inputs_gyro_dom, inputs_gyro_ndom], axis = 2)
    if params.f_strategy == 'late':
        return fuse_late(params, inputs_accel, inputs_gyro, is_training)
    elif params.f_strategy == 'early':
        return fuse_early(params, inputs_accel, inputs_gyro, is_training)
    else:
        raise RuntimeError('{0} is not supported in fusion.py'.format(params.f_strategy))


def ag1(params, inputs, is_training):
    if params.f_strategy == 'late':
        seq_pool, inputs_accel, inputs_gyro = ag(params, inputs, is_training)
        return seq_pool, inputs_accel + inputs_gyro
    elif params.f_strategy == 'early':
        return ag(params, inputs, is_training)

def ag2(params, inputs, is_training):
    if params.f_strategy == 'late':
        seq_pool, inputs_accel, inputs_gyro = ag(params, inputs, is_training)
        return seq_pool, inputs_accel + inputs_gyro + inputs_gyro
    elif params.f_strategy == 'early':
        raise RuntimeError('{0} is not supported in fusion.py'.format(params.f_strategy))
    
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
        elif self.params.fusion == 'dom_ndom':
            if self.params.f_mode == 'dnd1': 
                return dnd1(self.params, inputs, is_training)