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
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--dataset_train', type=str)
    parser.add_argument('--dataset_val', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_pics', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


FLAGS, unparsed = parse_args()
number_of_classes=21

log_folder = FLAGS.train_dir
eval_folder = FLAGS.eval_dir
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)

is_training_placeholder = tf.placeholder(tf.bool)

# Define input
image_tensor_train, orig_img_tensor_train, annotation_tensor_train = inputs(FLAGS.dataset_train, train=True, batch_size=FLAGS.batch_size, num_epochs=1e4)
image_tensor_val, orig_img_tensor_val, annotation_tensor_val = inputs(FLAGS.dataset_val, train=False, num_epochs=1e4)

image_tensor, orig_img_tensor, annotation_tensor = tf.cond(is_training_placeholder,
                                                           true_fn=lambda: (image_tensor_train, orig_img_tensor_train, annotation_tensor_train),
                                                           false_fn=lambda: (image_tensor_val, orig_img_tensor_val, annotation_tensor_val))

# Define loss
upsampled_logits = vgg16_fcn_net(image_tensor,number_of_classes,is_training=is_training_placeholder)
lbl_onehot = tf.one_hot(annotation_tensor, number_of_classes)
cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=upsampled_logits,
                                                          labels=lbl_onehot)

cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(cross_entropies, axis=-1))

# Tensor to get the final prediction for each pixel -- pay
# attention that we don't need softmax in this case because
# we only need the final decision. If we also need the respective
# probabilities we will have to apply softmax.
pred = tf.argmax(upsampled_logits, axis=3)

probabilities = tf.nn.softmax(upsampled_logits)

# Add summary op for the loss -- to be able to see it in
# tensorboard.
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)


global_step,train_step=optimizer(cross_entropy_loss,FLAGS.learning_rate,
    global_step = tf.train.get_or_create_global_step())

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
        gs, _ = sess.run([global_step, train_step],feed_dict={is_training_placeholder: True})
        if gs % 10 == 0:
            gs, loss, summary_string = sess.run([global_step, cross_entropy_loss, 
                merged_summary_op], feed_dict={is_training_placeholder: True})
            logging.info("step {0} Current Loss: {1} ".format(gs, loss))
            end = time.time()
            logging.info("[{0:.2f}] imgs/s".format(10 * FLAGS.batch_size / (end - start)))
            start = end

            summary_string_writer.add_summary(summary_string, gs)

            if gs % 100 == 0:
                save_path = saver.save(sess, os.path.join(log_folder, "model.ckpt"), global_step=gs)
                logging.info("Model saved in file: %s" % save_path)
            
            if gs % 200 == 0:
                logging.debug("validation generated at step [{0}]".format(gs))

                for idx in range(FLAGS.num_pics):
                    val_pred, val_orig, val_annot, val_poss = sess.run([pred, orig_img_tensor, annotation_tensor, probabilities],
                                                                             feed_dict={is_training_placeholder: False})
                    crf_ed = perform_crf(val_orig, val_poss,number_of_classes)
                    
                    save_image(eval_folder,val_orig,val_annot,val_pred,crf_ed,
                        prefix='{0}_{1}_'.format(gs,idx))
        
    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(sess, os.path.join(log_folder, "model.ckpt"), global_step=gs)
    logging.info("Model saved in file: %s" % save_path)

summary_string_writer.close()
