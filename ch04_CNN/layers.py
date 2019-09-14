import tensorflow as tf


def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def conv_layer(input, shape):
    W = weight_variables(shape)
    b = bias_variables([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    input_size = int(input.get_shape()[1])
    W = weight_variables([input_size, size])
    b = bias_variables([size])
    return tf.matmul(input, W) + b