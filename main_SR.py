# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 21:07:19 2017

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
from Loader import Loader
import re
from SuperRes import SuperRes
from scipy.misc import imresize, toimage
from PIL import Image
from scipy import signal, ndimage

from layer_def import relu_, res_, deconv_, conv_, fc_
import os
import numpy as np


def list_files(path):
    # returns a list of names (with extension, without full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files
    
def main(_):

    #setting a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    #sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#    with tf.Graph().as_default():
    sess = tf.Session(config=config)
#    print 'Path : ', cfg.IMAGES
    file_list = glob.glob(cfg.IMAGES)
#    print 'files_list : ', file_list
    cfg.NUM_IMAGES = len(file_list)
    print 'NUM_IMAGES : ', cfg.NUM_IMAGES
    cfg.NUM_TRAIN_IMAGES = int(cfg.NUM_IMAGES * cfg.TRAIN_RATIO)
    cfg.NUM_VAL_IMAGES = int(cfg.NUM_IMAGES * cfg.VAL_RATIO)
    train_images = file_list[:cfg.NUM_TRAIN_IMAGES]
    val_images = file_list[cfg.NUM_TRAIN_IMAGES:cfg.NUM_TRAIN_IMAGES + cfg.NUM_VAL_IMAGES]
#    print 'val images : ', val_images

    #load training & validation set
    loader = Loader(train_images, val_images)

    model = SuperRes(sess, loader)
    model.train_model()

    # Test a trained model
#    test_path='/dccstor/aurmmaret1/sajini-code/Kaggle-Data/SRGAN/Vesset_test_set/original_1024'
    test_path='/dccstor/aurmmaret1/sajini-code/Kaggle-Data/SRGAN/Testing_set'
    TEST_IMGS = list_files(test_path)
#    OUT_FILE = cfg.OUTPUT_DIR + "test_{i}"
    with open("SSIM_scale16.tsv", "w") as record_file:    
        for i, img in enumerate(TEST_IMGS):
#            out_file = OUT_FILE.replace("{i}", str(i))
            out_file = cfg.OUTPUT_DIR + img[0:-5]
            im1 = os.path.join(test_path, img)
            ssim, mse, psnr = model.predict(im1, out_file, init_vars=False)
            record_file.write(img + "  " + str(ssim) + "  " + str(mse) + "  " + str(psnr) + "\n")
    sess.close()

        

if __name__ == '__main__':
    tf.app.run()