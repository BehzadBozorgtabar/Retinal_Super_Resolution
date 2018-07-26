# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 21:45:24 2017

@author: Behzad
"""
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import glob
import argparse
import logging
import os
import sys
import config as cfg
import re
from scipy.misc import imresize, toimage
from PIL import Image
from scipy import signal, ndimage

#from blocks import relu_block, res_block, deconv_block, conv_block, dense_block


class Loader(object):
    def __init__(self, train_images, val_images):
#        print 'images : ', train_images
        self.q_train = tf.train.string_input_producer(train_images)
        self.q_val = tf.train.string_input_producer(val_images)
        cfg.NUM_TRAIN_BATCHES = len(train_images) // cfg.BATCH_SIZE
        cfg.NUM_VAL_BATCHES = len(val_images) // cfg.BATCH_SIZE


    def _get_pipeline(self, q):
        reader = tf.WholeFileReader()
        key, value = reader.read(q)
        print 'val', value
        raw_img = tf.image.decode_jpeg(value, channels=cfg.NUM_CHANNELS)
        print 'raw_img', raw_img
        my_img = tf.random_crop(raw_img, [cfg.HR_HEIGHT, cfg.HR_WIDTH, cfg.NUM_CHANNELS],
                seed=cfg.RANDOM_SEED)
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * cfg.BATCH_SIZE
        batch = tf.train.shuffle_batch([my_img], batch_size=cfg.BATCH_SIZE, capacity=capacity,
                min_after_dequeue=min_after_dequeue, seed=cfg.RANDOM_SEED)
        small_batch = tf.image.resize_bicubic(batch, [cfg.LR_HEIGHT, cfg.LR_WIDTH])
        return (small_batch, batch)

    def batch(self):
        return (self._get_pipeline(self.q_train),
                self._get_pipeline(self.q_val))