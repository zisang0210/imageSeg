#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import cv2
import tensorflow as tf

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (create_pairwise_bilateral,
                              create_pairwise_gaussian, unary_from_softmax)
import vgg
from dataset import inputs
from utils import (bilinear_upsample_weights, grayscale_to_voc_impl)
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

slim = tf.contrib.slim

def perform_crf(image, probabilities,number_of_classes):

    image = image.squeeze()
    softmax = probabilities.squeeze().transpose((2, 0, 1))

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    unary = unary_from_softmax(softmax)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(image.shape[0] * image.shape[1], number_of_classes)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                      img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return res

def upsample(feature_map,filter_name,upsample_factor,pool_feature,pool_scope,number_of_classes):

    with tf.variable_scope('vgg_16/fc8'):
        aux_logits = slim.conv2d(pool_feature, number_of_classes, [1, 1],
                                     activation_fn=None,
                                     weights_initializer=tf.zeros_initializer,
                                     scope=pool_scope)

    upsample_filter_tensor = bilinear_upsample_weights(upsample_factor,number_of_classes,filter_name)

    upsampled_feature_map = tf.nn.conv2d_transpose(feature_map, upsample_filter_tensor,
                                              output_shape=tf.shape(aux_logits),
                                              strides=[1, upsample_factor, upsample_factor, 1],
                                              padding='SAME')

    return upsampled_feature_map + aux_logits

def vgg16_fcn_net(image_tensor,number_of_classes,is_training=True,upsample_factor = 8):
    # tf.reset_default_graph()

    # Define the model that we want to use -- specify to use only two classes at the last layer
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(image_tensor,
                                        num_classes=number_of_classes,
                                        is_training=is_training,
                                        spatial_squeeze=False,
                                        fc_conv_padding='SAME')

    downsampled_logits_shape = tf.shape(logits)

    img_shape = tf.shape(image_tensor)

    # Calculate the ouput size of the upsampled tensor
    # The shape should be batch_size X width X height X num_classes
    upsampled_logits_shape = tf.stack([
                                      downsampled_logits_shape[0],
                                      img_shape[1],
                                      img_shape[2],
                                      downsampled_logits_shape[3]
                                      ])

    # Perform the upsampling x2
    upsampled_logits=upsample(logits,'vgg_16/fc8/t_conv_x2',2,
        end_points['vgg_16/pool4'],'conv_pool4',number_of_classes)
    # Perform the upsampling x2
    upsampled_logits=upsample(upsampled_logits,'vgg_16/fc8/t_conv_x2_x2',2,
        end_points['vgg_16/pool3'],'conv_pool3',number_of_classes)
    # Perform the upsampling x8
    upsample_filter_tensor_x8 = bilinear_upsample_weights(upsample_factor,
                                                       number_of_classes,
                                                       'vgg_16/fc8/t_conv_x8')
    upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x8,
                                              output_shape=upsampled_logits_shape,
                                              strides=[1, upsample_factor, upsample_factor, 1],
                                              padding='SAME')

    return upsampled_logits

def vgg16_fcn_loss(image_tensor,annotation_tensor,number_of_classes):
    upsampled_logits = vgg16_fcn_net(image_tensor,number_of_classes)
    lbl_onehot = tf.one_hot(annotation_tensor, number_of_classes)
    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=upsampled_logits,
                                                              labels=lbl_onehot)

    cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(cross_entropies, axis=-1))
    # Add summary op for the loss -- to be able to see it in
    # tensorboard.
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    return cross_entropy_loss
    
# Get the final prediction for each pixel -- pay
# attention that we don't need softmax in this case because
# we only need the final decision. If we also need the respective
# probabilities we will have to apply softmax.   
def vgg16_fcn_pred(image_tensor_val,number_of_classes):
    logits = vgg16_fcn_net(image_tensor_val,number_of_classes,is_training=False)

    pred = tf.argmax(logits, axis=3)

    probabilities = tf.nn.softmax(logits)
    return pred,probabilities

# Here we define an optimizer and put all the variables
# that will be created under a namespace of 'adam_vars'.
# This is done so that we can easily access them later.
# Those variables are used by adam optimizer and are not
# related to variables of the vgg model.

# We also retrieve gradient Tensors for each of our variables
# This way we can later visualize them in tensorboard.
# optimizer.compute_gradients and optimizer.apply_gradients
# is equivalent to running:
# train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy_loss)
def optimizer(cross_entropy_loss,lr,global_step):
    with tf.variable_scope("adam_vars"):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        gradients = optimizer.compute_gradients(loss=cross_entropy_loss)

        for grad_var_pair in gradients:

            current_variable = grad_var_pair[1]
            current_gradient = grad_var_pair[0]

            # Relace some characters from the original variable name
            # tensorboard doesn't accept ':' symbol
            gradient_name_to_save = current_variable.name.replace(":", "_")

            # Let's get histogram of gradients for each layer and
            # visualize them later in tensorboard
            tf.summary.histogram(gradient_name_to_save, current_gradient)

        train_step = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)
    return global_step,train_step

# Now we define a function that will load the weights from VGG checkpoint
# into our variables when we call it. We exclude the weights from the last layer
# which is responsible for class predictions. We do this because
# we will have different number of classes to predict and we can't
# use the old ones as an initialization.
def restore(sess,saver,vgg_checkpoint_path,log_folder):
    # Create the log folder if doesn't exist yet
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    checkpoint_path = tf.train.latest_checkpoint(log_folder)

    if checkpoint_path:
        restore_from_log(sess,saver,checkpoint_path)
    else:
        restore_from_ckpt(sess,saver,vgg_checkpoint_path)

def restore_from_log(sess,saver,checkpoint_path):
    logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % checkpoint_path)
    variables_to_restore = slim.get_model_variables()

    saver.restore(sess, checkpoint_path)

    logging.info('checkpoint restored from [{0}]'.format(checkpoint_path))

def restore_from_ckpt(sess,saver,vgg_checkpoint_path):
    vgg_except_fc8_weights = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'adam_vars'])

    # Here we get variables that belong to the last layer of network.
    # As we saw, the number of classes that VGG was originally trained on
    # is different from ours -- in our case it is only 2 classes.
    vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])

    adam_optimizer_variables = slim.get_variables_to_restore(include=['adam_vars'])


    # Create an OP that performs the initialization of
    # values of variables to the values from VGG.
    read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(
    vgg_checkpoint_path,
    vgg_except_fc8_weights)

    # Initializer for new fc8 weights -- for two classes.
    vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)

    # Initializer for adam variables
    optimization_variables_initializer = tf.variables_initializer(adam_optimizer_variables)

    sess.run(vgg_fc8_weights_initializer)
    sess.run(optimization_variables_initializer)

    read_vgg_weights_except_fc8_func(sess)
    logging.debug('value initialized...')

def save_image(eval_dir,val_orig,val_annot,val_pred,crf_ed,prefix=''):
    overlay = cv2.addWeighted(cv2.cvtColor(np.squeeze(val_orig), cv2.COLOR_RGB2BGR), 1, cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(crf_ed)), cv2.COLOR_RGB2BGR), 0.8, 0)

    cv2.imwrite(os.path.join(eval_dir, '{0}img.jpg'.format(prefix)), cv2.cvtColor(np.squeeze(val_orig), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(eval_dir, '{0}annotation.jpg'.format(prefix)),  cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(val_annot)), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(eval_dir, '{0}prediction.jpg'.format(prefix)),  cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(val_pred)), cv2.COLOR_RGB2BGR))        
    cv2.imwrite(os.path.join(eval_dir, '{0}prediction_crfed.jpg'.format(prefix)), cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(crf_ed)), cv2.COLOR_RGB2BGR))        
    cv2.imwrite(os.path.join(eval_dir, '{0}overlay.jpg'.format(prefix)), overlay)