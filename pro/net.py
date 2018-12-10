import numpy as np
import tensorflow as tf

image_size = 240
channels = 4
slices = 155
num_labels = 4

def inference(image_batch, train, regularizer):

     with tf.variable_scope('Encoder_cnn-pool-1'):









