# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 21:50:52 2017

@author: Behzad
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import glob
#import argparse
import logging
import os
import sys
import config as cfg
import re
from scipy.misc import imresize, toimage
from PIL import Image
from scipy import signal, ndimage

from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from skimage.measure import compare_psnr

from layer_def import relu_, res_, deconv_, conv_, fc_


class GAN(object):
    def __init__(self):
        self.g_images = tf.placeholder(tf.float32,
            [cfg.BATCH_SIZE, cfg.LR_HEIGHT, cfg.LR_WIDTH, cfg.NUM_CHANNELS])
        self.d_images = tf.placeholder(tf.float32,
            [cfg.BATCH_SIZE, cfg.HR_HEIGHT, cfg.HR_WIDTH, cfg.NUM_CHANNELS])
        self.is_training = tf.placeholder(tf.bool, shape=[])
        self.test_images = None

    def build_model(self, input_images=None):
        if input_images is not None:
            with tf.variable_scope("G", reuse=True) as scope:
                self.test_images = tf.placeholder(tf.float32, input_images.shape)
                scope.reuse_variables()
                self.G = self.generator(reuse=True)
        else:
            with tf.variable_scope("G"):
                self.G = self.generator(reuse=False)
                
            with tf.variable_scope("D") as scope:

                self.D = self.discriminator(self.d_images, reuse=False)
                scope.reuse_variables()
                self.D_ = self.discriminator(self.G, reuse=True)

            # Generator loss
            self.mse_loss = tf.reduce_mean(
                tf.squared_difference(self.d_images, self.G))

            self.g_gan_loss = (tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                    self.D_, tf.ones_like(self.D_))))

            self.g_loss = self.mse_loss + cfg.AD_LOSS_WEIGHT * self.g_gan_loss
            tf.scalar_summary('mse_loss', self.mse_loss)      
            tf.scalar_summary('g_gan_loss', self.g_gan_loss)
            tf.scalar_summary('g_loss', self.g_loss)

            # Discriminator loss
            self.d_loss_real = (tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                    self.D, tf.ones_like(self.D))))
            self.d_loss_fake = (tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                    self.D_, tf.zeros_like(self.D_))))

            self.d_loss = self.d_loss_real + self.d_loss_fake
            tf.scalar_summary('d_loss', self.d_loss)

            t_vars = tf.trainable_variables()

            self.d_vars = [var for var in t_vars if 'D/' in var.name]
            self.g_vars = [var for var in t_vars if 'G/' in var.name]


    def generator(self, reuse=False):

        with tf.variable_scope("g_conv1"):
            if self.test_images is not None:
                h = conv_(self.test_images, relu=True, reuse=reuse)
                print 'g_conv1 : ',h.get_shape()
            else:
                # noise = tf.random_normal(self.g_images.get_shape(), stddev=.03 * 255)
                h = self.g_images
                h = conv_(self.g_images, relu=True, reuse=reuse)
                print 'g_conv1 : ',h.get_shape()

        for i in range(1, 5):
            with tf.variable_scope("g_res" + str(i)):
                h = res_(h, self.is_training, reuse=reuse)
                print 'g_res : ',h.get_shape()

        with tf.variable_scope("g_deconv1"):
            h = deconv_(h)
            print 'g_deconv1 : ',h.get_shape()
        #------added    
        with tf.variable_scope("g_deconv2"):
            h = deconv_(h)
            print 'g_deconv2 : ',h.get_shape()
#            
        with tf.variable_scope("g_deconv3"):
            h = deconv_(h)
            print 'g_deconv3 : ',h.get_shape()
#            
        with tf.variable_scope("g_deconv4"):
            h = deconv_(h)
            print 'g_deconv4 : ',h.get_shape()
#
#        with tf.variable_scope("g_deconv5"):
#            h = deconv_(h)
#            print 'g_deconv5 : ',h.get_shape()
        # add more deconvolution layers if it is needed

        with tf.variable_scope("g_conv2"):
            h = conv_(h, output_channels=3, reuse=reuse)
            print 'g_conv2 : ',h.get_shape()
            
        return h



    def discriminator(self, inp, reuse=False):

        with tf.variable_scope("d_conv1"):
            h = conv_(inp, leaky_relu=True, reuse=reuse)
            print 'd_conv1 : ',h.get_shape()
#        with tf.variable_scope("d_conv12"):
#            h = conv_(h, leaky_relu=True, reuse=reuse)
#            print 'd_conv12 : ',h.get_shape()
#        with tf.variable_scope("d_conv13"):
#            h = conv_(inp, leaky_relu=True, reuse=reuse)

        with tf.variable_scope("d_conv2"):
            h = conv_(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, stride=2, reuse=reuse)
            print 'd_conv2 : ',h.get_shape()

        with tf.variable_scope("d_conv3"):
            h = conv_(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=128,
                reuse=reuse)
            print 'd_conv3 : ',h.get_shape()

        with tf.variable_scope("d_conv4"):
            h = conv_(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=128, stride=2,
                reuse=reuse)
            print 'd_conv4 : ',h.get_shape()

        with tf.variable_scope("d_conv5"):
            h = conv_(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=256, stride=1,
                reuse=reuse)
            print 'd_conv5 : ',h.get_shape()

        with tf.variable_scope("d_conv6"):
            h = conv_(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=256, stride=2,
                reuse=reuse)
            print 'd_conv6 : ',h.get_shape()

        with tf.variable_scope("d_conv7"):
            h = conv_(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=512, stride=1,
                reuse=reuse)
            print 'd_conv7 : ',h.get_shape()

        with tf.variable_scope("d_conv8"):
            h = conv_(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=512, stride=2,
                reuse=reuse)
            print 'd_conv8 : ',h.get_shape()

        with tf.variable_scope("d_dense1"):
            h = fc_(h, leaky_relu=True, sigmoid=False, output_size=1024)
            print 'd_dense1 : ',h.get_shape()

        with tf.variable_scope("d_dense2"):
            h = fc_(h, leaky_relu=False, sigmoid=True, output_size=1)
            print 'd_dense2 : ',h.get_shape()
        return h

class SuperRes(object):
    def __init__(self, sess, loader):
        #logging.info("Building Model.")
        print("Constructing Model")
        self.sess = sess
        self.loader = loader
        #self.train_batch, self.val_batch, self.test_batch = loader.batch()
        self.train_batch, self.val_batch = loader.batch()


        self.GAN = GAN()
        self.GAN.build_model()

        self.g_mse_optim = (tf.train.AdamOptimizer(cfg.LEARNING_RATE, beta1=cfg.BETA_1)
            .minimize(self.GAN.mse_loss, var_list=self.GAN.g_vars))
        self.d_optim = (tf.train.AdamOptimizer(cfg.LEARNING_RATE, beta1=cfg.BETA_1)
            .minimize(self.GAN.d_loss, var_list=self.GAN.d_vars))
        self.g_optim = (tf.train.AdamOptimizer(cfg.LEARNING_RATE, beta1=cfg.BETA_1)
            .minimize(self.GAN.g_loss, var_list=self.GAN.g_vars))

        batchnorm_updates = tf.get_collection(ops.GraphKeys.UPDATE_OPS)
        self.pretrain = tf.group(self.g_mse_optim, *batchnorm_updates)
        self.train = tf.group(self.d_optim, self.g_optim, *batchnorm_updates)

    def predict(self, input_name, output_name, init_vars=False):
        if init_vars == True:
            self._load_latest_checkpoint_or_initialize(tf.train.Saver())
        image = Image.open(input_name)
#        imageR = image.resize((1024, 1024), Image.NEAREST)
        hr = np.asarray(image, dtype=np.uint8)

        w = hr.shape[0] - hr.shape[0] % 4
        h = hr.shape[1] - hr.shape[1] % 4

        hr = hr[:w,:h]
        lr = imresize(hr, (w // cfg.r, h // cfg.r), interp='bicubic')

#        bicubic = imresize(lr, (cfg.r * lr.shape[0], cfg.r * lr.shape[1]), interp='bicubic')

        image = np.reshape(lr, (1,) + lr.shape)

        test_GAN = GAN()
        test_GAN.build_model(input_images=image)
        sr = self.sess.run(
            [test_GAN.G],
            feed_dict={
                test_GAN.test_images: image,
                test_GAN.is_training: False
        })
        sr = np.maximum(np.minimum(sr[0][0], 255.0), 0.0)

        #logging.info("SSIM - Bicubic %f, SR %f", ssim(bicubic, hr), ssim(sr, hr))
        ssim = compare_ssim(hr, sr.astype(np.uint8), multichannel=True)
        mse = compare_mse(hr, sr.astype(np.uint8))
        psnr = compare_psnr(hr, sr.astype(np.uint8))

        toimage(lr, cmin=0., cmax=255.).save(output_name + '_lr.jpeg')
#        toimage(bicubic, cmin=0., cmax=255.).save(output_name + '_bc.JPEG')
#        toimage(hr, cmin=0., cmax=255.).save(output_name + '_hr.jpeg')
        toimage(sr, cmin=0., cmax=255.).save(output_name + '_sr.jpeg')
        
        return ssim, mse, psnr


    def _load_latest_checkpoint_or_initialize(self, saver, attempt_load=True):
        if cfg.WEIGHTS:
            logging.info("Loading params from " + cfg.WEIGHTS)
            saver.restore(self.sess, cfg.WEIGHTS)
            return cfg.WEIGHTS
        ckpt_files = list(filter(lambda x: "meta" not in x, glob.glob(cfg.CHECKPOINT + "*")))
        if attempt_load and len(ckpt_files) > 0:
            ckpt_files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)])
            logging.info("Loading params from " + ckpt_files[-1])
            saver.restore(self.sess, ckpt_files[-1])
            return ckpt_files[-1]
        else:
            logging.info("Initializing parameters")
            self.sess.run(tf.initialize_all_variables())
            return ""

    def _pretrain(self):
#        merged_summary_op = tf.summary.merge_all()
#        summary_writer = tf.summary.FileWriter('/home/hsajini/RetinaProject/venv-keras-tf/Super-resolution/logs', session.graph)
        
        lr, hr = self.sess.run(self.train_batch)
        summary, _, loss = self.sess.run(
            [self.merged, self.pretrain, self.GAN.mse_loss],
            feed_dict={
                self.GAN.g_images: lr,
                self.GAN.d_images: hr,
                self.GAN.is_training: True
        })
        
#        summary_writer.add_summary(summary, global_step.eval(session=sess))        
        return summary, loss

    def _train(self):
        """
        Returns (summary, mse_loss, g_ad_loss, g_loss, d_loss_real, d_loss_fake, d_loss)
        """
        lr, hr = self.sess.run(self.val_batch)
        res = self.sess.run(
            [self.train, self.merged,
             self.GAN.g_loss, self.GAN.mse_loss, self.GAN.g_gan_loss,
             self.GAN.d_loss, self.GAN.d_loss_real, self.GAN.d_loss_fake],
            feed_dict={
                self.GAN.g_images: lr,
                self.GAN.d_images: hr,
                self.GAN.is_training: True
        })

        return res[1:]

    def _val(self):
        """
        Returns (summary, mse_loss, g_ad_loss, g_loss, d_loss_real, d_loss_fake, d_loss)
        """
        lr, hr = self.sess.run(self.train_batch)
        res = self.sess.run(
            [self.merged,
             self.GAN.g_loss, self.GAN.mse_loss, self.GAN.g_gan_loss,
             self.GAN.d_loss, self.GAN.d_loss_real, self.GAN.d_loss_fake],
            feed_dict={
                self.GAN.g_images: lr,
                self.GAN.d_images: hr,
                self.GAN.is_training: False
        })

        return res

    def _test(self):
        """
        Returns (summary, mse_loss, g_ad_loss, g_loss, d_loss_real, d_loss_fake, d_loss)
        """
        lr, hr = self.sess.run(self.test_batch)
        res = self.sess.run(
            [self.merged,
             self.GAN.g_loss, self.GAN.mse_loss, self.GAN.g_ad_loss,
             self.GAN.d_loss, self.GAN.d_loss_real, self.GAN.d_loss_fake],
            feed_dict={
                self.GAN.g_images: lr,
                self.GAN.d_images: hr,
                self.GAN.is_training: False
        })

        return res

    def _print_losses(self, losses, count):
        avg_losses = [x / count for x in losses]
        logging.info("G Loss: %f, MSE Loss: %f, Ad Loss: %f"
                % (avg_losses[0], avg_losses[1], avg_losses[2]))
        logging.info("D Loss: %f, Real Loss: %f, Fake Loss: %f"
                % (avg_losses[3], avg_losses[4], avg_losses[5]))

    def train_model(self):
        logging.info("Running on %d images" % (cfg.NUM_IMAGES,))
        print("Running on %d images" % (cfg.NUM_IMAGES,))
        self.merged = tf.merge_all_summaries()
        self.pre_train_writer = tf.train.SummaryWriter(os.path.join(cfg.LOGS_DIR, 'pretrain'),
                self.sess.graph)
        self.train_writer = tf.train.SummaryWriter(os.path.join(cfg.LOGS_DIR, 'train'),
                self.sess.graph)
        self.val_writer = tf.train.SummaryWriter(os.path.join(cfg.LOGS_DIR, 'val'),
                self.sess.graph)
        saver = tf.train.Saver(max_to_keep=None)
        ckpt = self._load_latest_checkpoint_or_initialize(saver, attempt_load=cfg.USE_CHECKPOINT)
        match = re.search(r'\d+$', ckpt)
        done_batch = int(match.group(0)) if match else 0

        sess = self.sess
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if "adversarial" not in ckpt:
            # Pretrain
            logging.info("Begin Pre-Training")
            print ("Begin Pre-Training")
            ind = 0
            for epoch in range(done_batch + 1, cfg.NUM_PRETRAIN_EPOCHS + 1):
                logging.info("Pre-Training Epoch: %d" % (epoch,))
                loss_sum = 0
                print epoch
                for batch in range(cfg.NUM_TRAIN_BATCHES):
                    summary, loss = self._pretrain()
                    self.pre_train_writer.add_summary(summary, ind)
                    loss_sum += loss
                    ind += 1
                logging.info("Epoch MSE Loss: %f" % (loss_sum / cfg.NUM_TRAIN_BATCHES,))

                if epoch % 4 == 0:
                #if epoch % 2 == 0:

                    logging.info("Saving Checkpoint")
                    saver.save(sess, cfg.CHECKPOINT + str(epoch))
            done_batch = 0
        else:
            logging.info("Skipping Pre-Training")

        logging.info("Begin Training")
        print ("Begin Training")
        # Adversarial training
        ind = 0
        if not cfg.PRETRAIN_ONLY:
            for epoch in range(done_batch + 1, cfg.NUM_TRAIN_EPOCHS + 1):
                logging.info("Training Epoch: %d" % (epoch,))
                losses = [0 for _ in range(6)]
                for batch in range(cfg.NUM_TRAIN_BATCHES):
                    res = self._train()
                    self.train_writer.add_summary(res[0], ind)
                    losses = [x + y for x, y in zip(losses, res[1:])]
                    ind += 1
                    if ind % 100 == 0:
                        self._print_losses(losses, 100)
                        losses = [0 for _ in range(6)]

                # Validation
                losses = [0 for _ in range(6)]
                for batch in range(cfg.NUM_VAL_BATCHES):
                    res = self._val()
                    self.val_writer.add_summary(res[0], ind)
                    losses = [x + y for x, y in zip(losses, res[1:])]
                    ind += 1

                logging.info("Epoch Validation Losses")
                self._print_losses(losses, cfg.NUM_VAL_BATCHES)

		logging.info("Saving Checkpoint (Adversarial)")
                saver.save(sess, cfg.CHECKPOINT + "_adversarial" + str(epoch))

        coord.request_stop()
        coord.join(threads)

    def test_model(self):
        val_writer = tf.train.SummaryWriter(join(cfg.LOGS_DIR, 'test'), self.sess.graph)

        with self.sess as sess:
            logging.info("Begin Testing")
            # Test
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            ind = 0
            for batch in range(cfg.NUM_TEST_BATCHES):
                lr, hr = sess.run(self.test_batch)
                res = self._test()
                test_writer.add_summary(res[0], ind)
                losses = [x + y for x, y in zip(losses, res[1:])]
                ind += 1

            logging.info("Test Losses")
            self._print_losses(losses, cfg.NUM_TEST_BATCHES)

            coord.request_stop()
            coord.join(threads)