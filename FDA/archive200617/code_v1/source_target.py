import tensorflow as tf
import numpy as np
import argparse
import glob
import os
import scipy.io
import math
import multiprocessing as mp
import time
from sklearn.metrics import roc_auc_score

pi = math.pi

from load_data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
source = 'data/CLB'
target = 'data/FDA'
source_model_name = 'noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k'
nb_train = 85000

# load target images
_, X_val, X_tst, _, y_val, y_tst = load_target(dataset = 'total', train = nb_train)
X_val, X_tst = (X_val-np.min(X_val))/(np.max(X_val)-np.min(X_val)), (X_tst-np.min(X_tst))/(np.max(X_tst)-np.min(X_tst))
X_val, X_tst = np.expand_dims(X_val, axis = 3), np.expand_dims(X_tst, axis = 3)
y_val, y_tst = y_val.reshape(-1,1), y_tst.reshape(-1,1)

# build the graph
# source_model_file = os.path.join(source, source_model_name, 'source_weights.h5')
# imported_graph = tf.train.import_meta_graph(os.path.join(source, source_model_name, 'source-best.meta'))
# def weight_variable(shape, name):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial, name=name)
# 
# def bias_variable(shape, name):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial, name=name)
# 
# def conv2d(x, w):
#     return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
# 
# def lrelu(x, alpha=0.2):
#   return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
# 
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding="SAME")
# 
# with tf.name_scope('input'):
#     x = tf.placeholder("float", shape=[None, 109,109, 1])
#     y_ = tf.placeholder("float", shape=[None, 1])

# bn_training = True
# with tf.name_scope('generator'):
#     W_conv1 = weight_variable([5, 5, 1, 32], 'conv1_weight')
#     b_conv1 = bias_variable([32], 'conv1_bias')
#     h_bn1 = tf.layers.batch_normalization(conv2d(x, W_conv1)+b_conv1, training = bn_training)
#     h_conv1 = lrelu(h_bn1)
#     h_pool1 = max_pool_2x2(h_conv1)
# 
#     W_conv2 = weight_variable([5, 5, 32, 32], 'conv2_weight')
#     b_conv2 = weight_variable([32], 'conv2_bias')
#     h_bn2 = tf.layers.batch_normalization(conv2d(h_pool1, W_conv2) + b_conv2, training = bn_training)
#     h_conv2 = lrelu(h_bn2)
# #     h_conv2 = lrelu(conv2d(h_pool1, W_conv2) + b_conv2)
#     h_pool2 = max_pool_2x2(h_conv2)
# 
#     W_conv3 = weight_variable([5, 5, 32, 32], 'conv3_weight')
#     b_conv3 = weight_variable([32], 'conv3_bias')
#     h_bn3 = tf.layers.batch_normalization(conv2d(h_pool2, W_conv3) + b_conv3, training = bn_training)
#     h_conv3 = lrelu(h_bn3)
# #     h_conv3 = lrelu(conv2d(h_pool2, W_conv3) + b_conv3)
#     h_pool3 = max_pool_2x2(h_conv3)
# 
#     h_pool3_flat = tf.reshape(h_pool3, [-1, 14*14*32])
#     W_fc1 = weight_variable([14*14*32, 1], 'fc1_weight')
#     b_fc1 = bias_variable([1], 'fc1_bias')
#     pred_logit = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
# 
# saver = tf.train.Saver()
saver = tf.train.import_meta_graph(os.path.join(source, source_model_name, 'source-best.meta'))
graph = tf.get_default_graph()
x = graph.get_tensor_by_name('input/Placeholder:0')
logit = graph.get_tensor_by_name('generator/add_3:0')

with tf.Session() as sess:
	# restore the saved vairable
	saver.restore(sess, os.path.join(source, source_model_name, 'source-best'))
	print(os.path.join(source, source_model_name, 'source-best'))
	# prediction
# 	test_logit = pred_logit.eval(session=sess,feed_dict={x:X_tst})
	test_logit = logit.eval(session=sess,feed_dict={x:X_tst})
	test_stat = np.exp(test_logit)
	test_AUC = roc_auc_score(y_tst, test_stat)

# 	val_logit = pred_logit.eval(session=sess,feed_dict={x:X_val})
	val_logit = logit.eval(session=sess,feed_dict={x:X_val})
	val_stat = np.exp(val_logit)
	val_AUC = roc_auc_score(y_val, val_stat)

	print('>>>> Source: {} To Target: {} >>>>'.format(source, target))
	print(' -AUC: valid {0:.4f}, test {1:.4f}\n'.format(val_AUC, test_AUC))	

	# save result
	with open(source+'/'+source_model_name+'/source_to_target.txt', 'w+') as f:
		f.write('>>>> Source {} To Target {} >>>>'.format(source, target))
		f.write(' -AUC: valid {0:.4f}, test {1:.4f}\n'.format(val_AUC, test_AUC))