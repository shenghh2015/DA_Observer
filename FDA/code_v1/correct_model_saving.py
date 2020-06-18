import numpy as np
import os
import glob

import tensorflow as tf


gpu_num = 6
lr = 1e-5
batch_size = 400
bn_training = True

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

source = 'data/CLB'
target = 'data/FDA'
source_model_name = 'noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k'
source_model_file = os.path.join(source, source_model_name, 'source-best')

DA = 'data/{}-{}'.format(os.path.basename(source), os.path.basename(target))
base_model_folder = os.path.join(DA, source_model_name)

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

def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return [d[shuffle_index] for d in data]

with tf.name_scope('input'):
    x = tf.placeholder("float", shape=[None, 109,109, 1])
    y_ = tf.placeholder("float", shape=[None, 1])

with tf.name_scope('generator'):
    W_conv1 = weight_variable([5, 5, 1, 32], 'conv1_weight')
    b_conv1 = bias_variable([32], 'conv1_bias')
    h_bn1 = tf.layers.batch_normalization(conv2d(x, W_conv1)+b_conv1, training = bn_training)
    h_conv1 = lrelu(h_bn1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 32], 'conv2_weight')
    b_conv2 = weight_variable([32], 'conv2_bias')
    h_bn2 = tf.layers.batch_normalization(conv2d(h_pool1, W_conv2) + b_conv2, training = bn_training)
    h_conv2 = lrelu(h_bn2)
#     h_conv2 = lrelu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([5, 5, 32, 32], 'conv3_weight')
    b_conv3 = weight_variable([32], 'conv3_bias')
    h_bn3 = tf.layers.batch_normalization(conv2d(h_pool2, W_conv3) + b_conv3, training = bn_training)
    h_conv3 = lrelu(h_bn3)
#     h_conv3 = lrelu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_pool3_flat = tf.reshape(h_pool3, [-1, 14*14*32])
    W_fc1 = weight_variable([14*14*32, 1], 'fc1_weight')
    b_fc1 = bias_variable([1], 'fc1_bias')
    pred_logit = tf.matmul(h_pool3_flat, W_fc1) + b_fc1

def get_variable(name):
	return [v for v in tf.global_variables() if v.name == name][0]

variable_direc = {'base_conv1_weight':W_conv1, 'base_conv1_bias':b_conv1, 
				  'base_bn1_gamma':get_variable('batch_normalization/gamma:0'), 'base_bn1_beta':get_variable('batch_normalization/beta:0'),
				  'base_conv2_weight':W_conv2, 'base_conv2_bias':b_conv2,
				  'base_bn2_gamma':get_variable('batch_normalization_1/gamma:0'), 'base_bn2_beta':get_variable('batch_normalization_1/beta:0'), 
				  'base_conv3_weight':W_conv3, 'base_conv3_bias':b_conv3,
				  'base_bn3_gamma':get_variable('batch_normalization_2/gamma:0'), 'base_bn3_beta':get_variable('batch_normalization_2/beta:0'),
				    'base_fc1_weight':W_fc1, 'base_fc1_bias':b_fc1
				  }

saver_all = tf.train.Saver()
saver = tf.train.Saver(variable_direc)

from load_data import *
# load source data
nb_source = 100000
noise = 2.0
sig_rate = 0.035
Xs_trn, Xs_val, Xs_tst, _, ys_val, ys_tst = load_source(train = nb_source, sig_rate = sig_rate)
Xs_trn, Xs_val, Xs_tst = np.random.RandomState(2).normal(Xs_trn, noise), np.random.RandomState(0).normal(Xs_val, noise), np.random.RandomState(1).normal(Xs_tst, noise)
Xs_trn, Xs_val, Xs_tst = (Xs_trn-np.min(Xs_trn))/(np.max(Xs_trn)-np.min(Xs_trn)), (Xs_val-np.min(Xs_val))/(np.max(Xs_val)-np.min(Xs_val)), (Xs_tst-np.min(Xs_tst))/(np.max(Xs_tst)-np.min(Xs_tst))
Xs_trn, Xs_val, Xs_tst = np.expand_dims(Xs_trn, axis = 3), np.expand_dims(Xs_val, axis = 3), np.expand_dims(Xs_tst, axis = 3)
ys_tst = ys_tst.reshape(-1,1)
# load target data
nb_target = 85000
Xt_trn, Xt_val, Xt_tst, _, yt_val, yt_tst = load_target(dataset = 'total', train = nb_target)
Xt_trn, Xt_val, Xt_tst = (Xt_trn-np.min(Xt_trn))/(np.max(Xt_trn)-np.min(Xt_trn)), (Xt_val-np.min(Xt_val))/(np.max(Xt_val)-np.min(Xt_val)), (Xt_tst-np.min(Xt_tst))/(np.max(Xt_tst)-np.min(Xt_tst))
Xt_trn, Xt_val, Xt_tst = np.expand_dims(Xt_trn, axis = 3), np.expand_dims(Xt_val, axis = 3), np.expand_dims(Xt_tst, axis = 3)
yt_tst = yt_tst.reshape(-1,1)

# sess = tf.Session()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
# 	saver_all.restore(sess, source_model_file)
# 	saver.restore(sess, source_model_file)
# 	saver.save(sess, base_model_folder+'/base_model')
	saver.restore(sess, base_model_folder+'/base_model')
	test_source_logit = pred_logit.eval(session=sess,feed_dict={x:Xs_tst})
	test_source_stat = np.exp(test_source_logit)
	test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
	test_target_logit = pred_logit.eval(session=sess,feed_dict={x:Xt_tst})
	test_target_stat = np.exp(test_target_logit)
	test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
	print('>>>>>> Check the Initial Source Model Loading <<<<<<')
	print('Source to Source:{0:.4f} '.format(test_source_AUC))
	print('Source to Target:{0:.4f}'.format(test_target_AUC))
	
bn_training = True
with tf.name_scope('source_model'):
	x_s = tf.placeholder("float", shape=[None, 109,109, 1])
	y_s = tf.placeholder("float", shape=[None, 1])
	W_conv1_s = weight_variable([5, 5, 1, 32], 'source_conv1_weight')
	b_conv1_s = bias_variable([32], 'source_conv1_bias')
	h_bn1_s = tf.layers.batch_normalization(conv2d(x_s, W_conv1_s)+b_conv1_s, training = bn_training, name='source_bn1')
	h_conv1_s = lrelu(h_bn1_s)
	h_pool1_s = max_pool_2x2(h_conv1_s)

	W_conv2_s = weight_variable([5, 5, 32, 32], 'source_conv2_weight')
	b_conv2_s = weight_variable([32], 'source_conv2_bias')
	h_bn2_s = tf.layers.batch_normalization(conv2d(h_pool1_s, W_conv2_s) + b_conv2_s, training = bn_training, name='source_bn2')
	h_conv2_s = lrelu(h_bn2_s)
	h_pool2_s = max_pool_2x2(h_conv2_s)

	W_conv3_s = weight_variable([5, 5, 32, 32], 'source_conv3_weight')
	b_conv3_s = weight_variable([32], 'source_conv3_bias')
	h_bn3_s = tf.layers.batch_normalization(conv2d(h_pool2_s, W_conv3_s) + b_conv3_s, training = bn_training, name='source_bn3')
	h_conv3_s = lrelu(h_bn3_s)
	h_pool3_s = max_pool_2x2(h_conv3_s)

	h_pool3_flat_s = tf.reshape(h_pool3_s, [-1, 14*14*32])
	W_fc1_s = weight_variable([14*14*32, 1], 'source_fc1_weight')
	b_fc1_s = bias_variable([1], 'source_fc1_bias')
	source_logit = tf.matmul(h_pool3_flat_s, W_fc1_s) + b_fc1_s

# for pl, sc in zip(players, scores): 
#     print ("Player :  %s     Score : %d" %(pl, sc)) 

source_vars_list = [v for v in tf.trainable_variables() if 'source_' in v.name]
source_vars_list = [v for v in tf.trainable_variables() if 'source_' in v.name]
saved_keys_list= ['base_conv1_weight', 'base_conv1_bias', 
				  'base_bn1_gamma', 'base_bn1_beta',
				  'base_conv2_weight', 'base_conv2_bias',
				  'base_bn2_gamma', 'base_bn2_beta', 
				  'base_conv3_weight', 'base_conv3_bias',
				  'base_bn3_gamma', 'base_bn3_beta',
				    'base_fc1_weight', 'base_fc1_bias']

source_direct = {}
for key, var in zip(saved_keys_list, source_vars_list):
	source_direct[key] = var

source_saver = tf.train.Saver(source_direct)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	source_saver.restore(sess, base_model_folder+'/base_model')
	test_source_logit = source_logit.eval(session=sess,feed_dict={x_s:Xs_tst})
	test_source_stat = np.exp(test_source_logit)
	test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
	test_target_logit = source_logit.eval(session=sess,feed_dict={x_s:Xt_tst})
	test_target_stat = np.exp(test_target_logit)
	test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
	print('>>>>>> Check the Initial Source Model Loading <<<<<<')
	print('Source to Source:{0:.4f} '.format(test_source_AUC))
	print('Source to Target:{0:.4f}'.format(test_target_AUC))
