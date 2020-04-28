import tensorflow as tf
import numpy as np
import os

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")

# def conv4_bn(x):
# 	with tf.name_scope('generator'):
# 		W_conv1 = weight_variable([5, 5, 1, 32], 'conv1_weight')
# 		b_conv1 = bias_variable([32], 'conv1_bias')
# 		h_bn1 = tf.layers.batch_normalization(conv2d(x, W_conv1)+b_conv1, training = bn_training)
# 		h_conv1 = lrelu(h_bn1)
# 
# 		W_conv2 = weight_variable([5, 5, 32, 32], 'conv2_weight')
# 		b_conv2 = weight_variable([32], 'conv2_bias')
# 		h_bn2 = tf.layers.batch_normalization(conv2d(h_conv1, W_conv2) + b_conv2, training = bn_training)
# 		h_conv2 = lrelu(h_bn2)
# 		h_pool2 = max_pool_2x2(h_conv2)
# 
# 		W_conv3 = weight_variable([5, 5, 32, 32], 'conv3_weight')
# 		b_conv3 = weight_variable([32], 'conv3_bias')
# 		h_bn3 = tf.layers.batch_normalization(conv2d(h_pool2, W_conv3) + b_conv3, training = bn_training)
# 		h_conv3 = lrelu(h_bn3)
# 
# 		W_conv4 = weight_variable([5, 5, 32, 32], 'conv4_weight')
# 		b_conv4 = weight_variable([32], 'conv4_bias')
# 		h_bn4 = tf.layers.batch_normalization(conv2d(h_conv3, W_conv4) + b_conv4, training = bn_training)
# 		h_conv4 = lrelu(h_bn4)
# 		h_pool4 = max_pool_2x2(h_conv4)
# 
# 		h_pool4_flat = tf.reshape(h_pool4, [-1, 28*28*32])
# 		W_fc1 = weight_variable([28*28*32, 1], 'fc1_weight')
# 		b_fc1 = bias_variable([1], 'fc1_bias')
# 		pred_logit = tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
# 
# 		variable_direc = {'base_conv1_weight':W_conv1, 'base_conv1_bias':b_conv1, 
# 					'base_bn1_gamma':get_variable('batch_normalization/gamma:0'), 'base_bn1_beta':get_variable('batch_normalization/beta:0'),
# 					'base_conv2_weight':W_conv2, 'base_conv2_bias':b_conv2,
# 					'base_bn2_gamma':get_variable('batch_normalization_1/gamma:0'), 'base_bn2_beta':get_variable('batch_normalization_1/beta:0'), 
# 					'base_conv3_weight':W_conv3, 'base_conv3_bias':b_conv3,
# 					'base_bn3_gamma':get_variable('batch_normalization_2/gamma:0'), 'base_bn3_beta':get_variable('batch_normalization_2/beta:0'),
# 					'base_conv4_weight':W_conv4, 'base_conv4_bias':b_conv4,
# 					'base_bn4_gamma':get_variable('batch_normalization_3/gamma:0'), 'base_bn4_beta':get_variable('batch_normalization_3/beta:0'),
# 					'base_fc1_weight':W_fc1, 'base_fc1_bias':b_fc1}
# 		saver = tf.train.Saver(variable_direc)
# 	
# 	return h_pool4_flat, h_pool4_flat, pred_logit, saver

# def _conv_bn_lrelu_pool(x, idx_cnn = 0, max_pool = True, bn = True):
# 	if idx_cnn == 0:
# 		kernel = [5, 5, 1, 32]
# 	else:
# 		kernel = [5, 5, 32, 32]
# 	W_conv = weight_variable(kernel, 'conv{}_weight'.format(name_scope, idx_cnn))
# 	b_conv = bias_variable([32], 'conv{}_bias'.format(name_scope, idx_cnn))
# 	if bn:
# 		h_conv = tf.layers.batch_normalization(conv2d(x, W_conv) + b_conv, training = True, name = 'bn{}'.format(name_scope, idx_cnn))
# 		h_conv = lrelu(h_conv)
# 	else:
# 		h_conv = lrelu(conv2d(x, W_conv)+b_conv)
# 	if max_pool:
# 		h_conv = max_pool_2x2(h_conv)
# 	return h_conv
# def _conv_bn_lrelu_pool(x, idx_cnn = 0, max_pool = True, bn = True):
# 	if idx_cnn == 0:
# 		kernel = [5, 5, 1, 32]
# 	else:
# 		kernel = [5, 5, 32, 32]
# 	W_conv = weight_variable(kernel, 'conv{}_weight'.format(name_scope, idx_cnn))
# 	b_conv = bias_variable([32], 'conv{}_bias'.format(name_scope, idx_cnn))
# 	if bn:
# 		h_conv = tf.layers.batch_normalization(conv2d(x, W_conv) + b_conv, training = True, name = 'bn{}'.format(name_scope, idx_cnn))
# 		h_conv = lrelu(h_conv)
# 	else:
# 		h_conv = lrelu(conv2d(x, W_conv)+b_conv)
# 	if max_pool:
# 		h_conv = max_pool_2x2(h_conv)
# 	return h_conv

l2_regularizer = tf.contrib.layers.l2_regularizer(1e-5)
def _conv_bn_lrelu_pool(x, pool = False, bn = True):
	_conv = tf.layers.conv2d(x, filters = 32, kernel_size = [5,5], strides=(1, 1), padding='same',
			kernel_initializer= 'truncated_normal', kernel_regularizer=l2_regularizer)
	if bn:
		_bn = tf.layers.batch_normalization(_conv, training = True)
	else:
		_bn = _conv
	_lrelu = tf.nn.leaky_relu(_bn)
	if pool:
		_out = max_pool_2x2(_lrelu)
	else:
		_out = _lrelu
	return _out

def conv_block(x, nb_cnn = 4, bn = True, scope_name = 'base'):
	with tf.variable_scope(scope_name):
		h = _conv_bn_lrelu_pool(x, pool = False, bn = bn)
		for i in range(1, nb_cnn):
			if i%2 == 1:
				pool = True
			else:
				pool = False
			h = _conv_bn_lrelu_pool(h, pool = pool, bn = bn)
	return h

# x = tf.placeholder("float", shape=[None, 109,109, 1])
# h = conv_block(x, nb_cnn = 4, bn = True, scope_name = 'base')

def dense_block(x, fc_layers = [128, 1], nb = True, scope_name = 'base'):
# 	shape = x.shape.as_list()[1:]
	with tf.variable_scope(scope_name):
		flat = tf.layers.flatten(x)
		h1 = tf.layers.dense(flat, fc_layers[0], kernel_regularizer=l2_regularizer)
		print(nb)
		if nb:
			h1 = tf.layers.batch_normalization(h1, training = True)
		h1 = tf.nn.leaky_relu(h1)
		h2 = tf.layers.dense(h1, fc_layers[1], kernel_regularizer = l2_regularizer)
	return h1, h2

### create network
def conv_classifier(x, nb_cnn = 4, fc_layers = [128,1],  bn = True, scope_name = 'base', reuse = False):
	with tf.variable_scope(scope_name, reuse = reuse):
		conv_net = conv_block(x, nb_cnn = nb_cnn, bn = bn, scope_name = 'conv')
		h, pred_logit = dense_block(conv_net, fc_layers = fc_layers, nb = bn, scope_name = 'classifier')

	return conv_net, h, pred_logit

def discriminator(x, nb_cnn = 2, fc_layers = [128, 1], bn = True, reuse = False):
	with tf.variable_scope('discriminator', reuse = reuse):
		if nb_cnn > 0:
			h = conv_block(x, nb_cnn = nb_cnn, bn = bn, scope_name = 'cov')
			_, pred_logit = dense_block(h, fc_layers = fc_layers, nb = bn, scope_name = 'fc')
		else:
			_, pred_logit = dense_block(x, fc_layers = fc_layers, nb = bn, scope_name = 'fc')
	return pred_logit
