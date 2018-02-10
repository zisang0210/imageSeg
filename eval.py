#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time
import numpy as np
import tensorflow as tf

from vgg16_fcn import *
from dataset import inputs
from utils import grayscale_to_voc_impl

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--dataset_val', type=str)
    parser.add_argument('--num_pics', type=int, default=10)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


FLAGS, unparsed = parse_args()
number_of_classes=21

# Define network
image_tensor, orig_img_tensor, annotation_tensor = inputs(FLAGS.dataset_val, train=False, num_epochs=1e4)

pred,probabilities = vgg16_fcn_pred(image_tensor,number_of_classes)

eval_dir = FLAGS.eval_dir
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

init_op = tf.global_variables_initializer()
init_local_op = tf.local_variables_initializer()

with tf.Session(config=sess_config) as sess:
    # Run the initializers.
    sess.run(init_op)
    sess.run(init_local_op)

    saver = tf.train.Saver(max_to_keep=5)
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
    restore_from_log(sess,saver,checkpoint_path)
    
    global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])

    logging.info('Evaluating at global_step = %d' % global_step)

    # start data reader
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    start = time.time()
    for i in range(FLAGS.num_pics):
        val_pred, val_orig_image, val_annot, val_poss = sess.run([pred, orig_img_tensor, annotation_tensor, probabilities])


        crf_ed = perform_crf(val_orig_image, val_poss,number_of_classes)

        save_image(eval_dir,val_orig_image,val_annot,val_pred,crf_ed,
            prefix='{0}_{1}_'.format(global_step,i))

    coord.request_stop()
    coord.join(threads)