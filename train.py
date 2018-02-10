#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time
import numpy as np
import tensorflow as tf

from vgg16_fcn import *
from dataset import inputs

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--dataset_train', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


FLAGS, unparsed = parse_args()
number_of_classes=21
# Define input
image_tensor, orig_img_tensor, annotation_tensor = inputs(FLAGS.dataset_train, train=True, batch_size=FLAGS.batch_size, num_epochs=1e4)

# Define loss
cross_entropy_loss = vgg16_fcn_loss(image_tensor,annotation_tensor,number_of_classes)

global_step,train_step=optimizer(cross_entropy_loss,FLAGS.learning_rate,
      global_step = tf.train.get_or_create_global_step())


log_folder = FLAGS.train_dir
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

init_op = tf.global_variables_initializer()
init_local_op = tf.local_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)

# Put all summary ops into one op. Produces string when
# you run it.
merged_summary_op = tf.summary.merge_all()

# Create the summary writer -- to write all the logs
# into a specified file. This file can be later read
# by tensorboard.
summary_string_writer = tf.summary.FileWriter(log_folder)

with tf.Session(config=sess_config) as sess:
    # Run the initializers.
    sess.run(init_op)
    sess.run(init_local_op)
    restore(sess,saver,FLAGS.checkpoint_path,log_folder)

    # start data reader
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    start = time.time()
    for i in range(FLAGS.max_steps):
        gs, _ = sess.run([global_step, train_step])
        if gs % 10 == 0:
            gs, loss, summary_string = sess.run([global_step, cross_entropy_loss, 
                merged_summary_op])
            logging.info("step {0} Current Loss: {1} ".format(gs, loss))
            end = time.time()
            logging.info("[{0:.2f}] imgs/s".format(10 * FLAGS.batch_size / (end - start)))
            start = end

            summary_string_writer.add_summary(summary_string, gs)

            if gs % 100 == 0:
                save_path = saver.save(sess, os.path.join(log_folder, "model.ckpt"), global_step=gs)
                logging.info("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(sess, os.path.join(log_folder, "model.ckpt"), global_step=gs)
    logging.info("Model saved in file: %s" % save_path)

summary_string_writer.close()
