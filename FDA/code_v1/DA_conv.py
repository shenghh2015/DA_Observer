import numpy as np
import os
import glob

import tensorflow as tf
import numpy as np
import argparse
import os
from sklearn.metrics import roc_auc_score
import scipy.io
import math
import multiprocessing as mp
import time

pi = math.pi
from natsort import natsorted
import time
import gc
from termcolor import colored 

import os
from load_data import *

def save_discr_epoch_idx(model,model_name,epoch_idx):
	generate_folder(model_name)
	# serialize model to YAML
	model_yaml = model.to_yaml()
	model_path = model_name+"/discr.yaml"
	if not os.path.exists(model_path):
		with open(model_path, "w") as yaml_file:
			yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights(model_name+"/discr_{}.h5".format(epoch_idx))


from natsort import natsorted

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

# plot and save the file
def plot_loss(model_name,loss,val_loss, file_name):
	generate_folder(model_name)
	f_out = file_name
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	start_idx = 0
	if len(loss)>start_idx:
		title = os.path.basename(os.path.dirname(file_name))
		fig = Figure(figsize=(8,6))
		ax = fig.add_subplot(1,1,1)
		ax.plot(loss[start_idx:],'b-',linewidth=1.3)
		ax.plot(val_loss[start_idx:],'r-',linewidth=1.3)
		ax.set_title(title)
		ax.set_ylabel('Loss')
		ax.set_xlabel('batches')
		ax.legend(['D-loss', 'G-loss'], loc='upper left')  
		canvas = FigureCanvasAgg(fig)
		canvas.print_figure(f_out, dpi=80)

def plot_auc_iterations(target_auc_list, val_auc_list, target_file_name):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	# fig = plt.figure()
	# plt.clf()
	# file_name = target_folder + '/target_auc.png'
	file_name = target_file_name
	ax = fig.add_subplot(111)
	ax.plot(target_auc_list)
	ax.plot(val_auc_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC')
	ax.legend(['Test','Val'])
	ax.set_xlim([0,len(target_auc_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def print_yellow(str):
	from termcolor import colored 
	print(colored(str, 'yellow'))

def print_red(str):
	from termcolor import colored 
	print(colored(str, 'red'))

def print_green(str):
	from termcolor import colored 
	print(colored(str, 'green'))

def print_block(symbol = '*', nb_sybl = 70):
	print_red(symbol*nb_sybl)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int)
parser.add_argument("--D_lr", type = float)
parser.add_argument("--G_lr", type = float)
parser.add_argument("--nD", type = int)
parser.add_argument("--nG", type = int)
parser.add_argument("--iters", type = int)
parser.add_argument("--bz", type = int)


args = parser.parse_args()
gpu_num = args.gpu
batch_size = args.bz
d_lr = args.D_lr
g_lr = args.G_lr
nb_steps = args.iters
nd_steps = args.nD
ng_steps = args.nG

if False:
	gpu_num = 6
	batch_size = 400
	dis_v = 1
	nb_dis = 5
	d_lr = 1e-5
	g_lr = 1e-6
	nb_steps = 1000
	nd_steps = 10
	ng_steps = 10

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
bn_training = True
# hyper-parameters
noise = 2.0
sig_rate = 0.035
source_model_name = 'noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k-v2'
# load source data
source = 'data/CLB'
target = 'data/FDA'
source_model_file = os.path.join(source, source_model_name, 'source-best')

# load source data
nb_source = 100000
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

DA = 'data/{}-{}'.format(os.path.basename(source), os.path.basename(target))
generate_folder(DA)
base_model_folder = os.path.join(DA, source_model_name)
generate_folder(base_model_folder)
# copy the source weight file to the DA_model_folder

DA_model_name = 'bz-{}-D_lr-{}-G_lr-{}-nD-{}-nG-{}-iter-{}'.format(batch_size, d_lr, g_lr, nd_steps, ng_steps, nb_steps)
DA_model_folder = os.path.join(base_model_folder, DA_model_name)
generate_folder(DA_model_folder)
os.system('cp -f {} {}'.format(source_model_file+'*', DA_model_folder))

# load the source encoder part from the source model
# source_model = globals()[source_func_name](input_shape = (109,109,1), kernel_initializer = initializer, bn = bn)
# source_model.load_weights(DA_model_folder+'/source_weights.h5', by_name = True)
# save_model(source_model, DA_model_folder)

# load the source model


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

bn_training = True
# with tf.name_scope('source_model'):
# 	x_s = tf.placeholder("float", shape=[None, 109,109, 1])
# 	y_s = tf.placeholder("float", shape=[None, 1])
# 	W_conv1_s = weight_variable([5, 5, 1, 32], 'source_conv1_weight')
# 	b_conv1_s = bias_variable([32], 'source_conv1_bias')
# 	h_bn1_s = tf.layers.batch_normalization(conv2d(x_s, W_conv1_s)+b_conv1_s, training = bn_training, name='source_bn1')
# 	h_conv1_s = lrelu(h_bn1_s)
# 	h_pool1_s = max_pool_2x2(h_conv1_s)
# 
# 	W_conv2_s = weight_variable([5, 5, 32, 32], 'source_conv2_weight')
# 	b_conv2_s = weight_variable([32], 'source_conv2_bias')
# 	h_bn2_s = tf.layers.batch_normalization(conv2d(h_pool1_s, W_conv2_s) + b_conv2_s, training = bn_training, name='source_bn2')
# 	h_conv2_s = lrelu(h_bn2_s)
# 	h_pool2_s = max_pool_2x2(h_conv2_s)
# 
# 	W_conv3_s = weight_variable([5, 5, 32, 32], 'source_conv3_weight')
# 	b_conv3_s = weight_variable([32], 'source_conv3_bias')
# 	h_bn3_s = tf.layers.batch_normalization(conv2d(h_pool2_s, W_conv3_s) + b_conv3_s, training = bn_training, name='source_bn3')
# 	h_conv3_s = lrelu(h_bn3_s)
# 	h_pool3_s = max_pool_2x2(h_conv3_s)
# 
# 	h_pool3_flat_s = tf.reshape(h_pool3_s, [-1, 14*14*32])
# 	W_fc1_s = weight_variable([14*14*32, 1], 'source_fc1_weight')
# 	b_fc1_s = bias_variable([1], 'source_fc1_bias')
# 	source_logit = tf.matmul(h_pool3_flat_s, W_fc1_s) + b_fc1_s
with tf.name_scope('source_model'):
	x_s = tf.placeholder("float", shape=[None, 109,109, 1])
	y_s = tf.placeholder("float", shape=[None, 1])
	W_conv1_s = weight_variable([5, 5, 1, 32], 'source_conv1_weight')
	b_conv1_s = bias_variable([32], 'source_conv1_bias')
	h_bn1_s = tf.layers.batch_normalization(conv2d(x_s, W_conv1_s)+b_conv1_s, training = bn_training, name='source_bn1')
	h_conv1_s = lrelu(h_bn1_s)
# 	h_pool1_s = max_pool_2x2(h_conv1_s)

	W_conv2_s = weight_variable([5, 5, 32, 32], 'source_conv2_weight')
	b_conv2_s = weight_variable([32], 'source_conv2_bias')
	h_bn2_s = tf.layers.batch_normalization(conv2d(h_conv1_s, W_conv2_s) + b_conv2_s, training = bn_training, name='source_bn2')
	h_conv2_s = lrelu(h_bn2_s)
	h_pool2_s = max_pool_2x2(h_conv2_s)

	W_conv3_s = weight_variable([5, 5, 32, 32], 'source_conv3_weight')
	b_conv3_s = weight_variable([32], 'source_conv3_bias')
	h_bn3_s = tf.layers.batch_normalization(conv2d(h_pool2_s, W_conv3_s) + b_conv3_s, training = bn_training, name='source_bn3')
	h_conv3_s = lrelu(h_bn3_s)
# 	h_pool3_s = max_pool_2x2(h_conv3_s)

	W_conv4_s = weight_variable([5, 5, 32, 32], 'source_conv4_weight')
	b_conv4_s = weight_variable([32], 'source_conv4_bias')
	h_bn4_s = tf.layers.batch_normalization(conv2d(h_conv3_s, W_conv4_s) + b_conv4_s, training = bn_training, name='source_bn4')
	h_conv4_s = lrelu(h_bn4_s)
	h_pool4_s = max_pool_2x2(h_conv4_s)

	h_pool4_flat_s = tf.reshape(h_pool4_s, [-1, 28*28*32])
	W_fc1_s = weight_variable([28*28*32, 1], 'source_fc1_weight')
	b_fc1_s = bias_variable([1], 'source_fc1_bias')
	source_logit = tf.matmul(h_pool4_flat_s, W_fc1_s) + b_fc1_s

def get_variable(name):
	return [v for v in tf.global_variables() if v.name == name][0]

source_vars_list = [v for v in tf.trainable_variables() if 'source_' in v.name]
# saved_keys_list= ['base_conv1_weight', 'base_conv1_bias', 
# 				  'base_bn1_gamma', 'base_bn1_beta',
# 				  'base_conv2_weight', 'base_conv2_bias',
# 				  'base_bn2_gamma', 'base_bn2_beta', 
# 				  'base_conv3_weight', 'base_conv3_bias',
# 				  'base_bn3_gamma', 'base_bn3_beta',
# 				    'base_fc1_weight', 'base_fc1_bias']
saved_keys_list= ['base_conv1_weight', 'base_conv1_bias', 
				  'base_bn1_gamma', 'base_bn1_beta',
				  'base_conv2_weight', 'base_conv2_bias',
				  'base_bn2_gamma', 'base_bn2_beta', 
				  'base_conv3_weight', 'base_conv3_bias',
				  'base_bn3_gamma', 'base_bn3_beta',
				  'base_conv4_weight', 'base_conv4_bias',
				  'base_bn4_gamma', 'base_bn4_beta',
				  'base_fc1_weight', 'base_fc1_bias'
				    ]
source_direct = {}
for key, var in zip(saved_keys_list, source_vars_list):
	source_direct[key] = var

source_saver = tf.train.Saver(source_direct)

#s_vars = [v for v in tf.trainable_variables() if 'source_' in v.name]
# s_trainer = tf.train.AdamOptimizer(0.0001).minimize(s_loss, var_list=s_vars)

bn_training = True
# with tf.name_scope('target_model'):
# 	x_t = tf.placeholder("float", shape=[None, 109,109, 1])
# 	y_t = tf.placeholder("float", shape=[None, 1])
# 	W_conv1_t = weight_variable([5, 5, 1, 32], 'target_conv1_weight')
# 	b_conv1_t = bias_variable([32], 'target_conv1_bias')
# 	h_bn1_t = tf.layers.batch_normalization(conv2d(x_t, W_conv1_t)+b_conv1_t, training = bn_training, name='target_bn1')
# 	h_conv1_t = lrelu(h_bn1_t)
# 	h_pool1_t = max_pool_2x2(h_conv1_t)
# 
# 	W_conv2_t = weight_variable([5, 5, 32, 32], 'target_conv2_weight')
# 	b_conv2_t = weight_variable([32], 'target_conv2_bias')
# 	h_bn2_t = tf.layers.batch_normalization(conv2d(h_pool1_t, W_conv2_t) + b_conv2_t, training = bn_training, name='target_bn2')
# 	h_conv2_t = lrelu(h_bn2_t)
# 	h_pool2_t = max_pool_2x2(h_conv2_t)
# 
# 	W_conv3_t = weight_variable([5, 5, 32, 32], 'target_conv3_weight')
# 	b_conv3_t = weight_variable([32], 'target_conv3_bias')
# 	h_bn3_t = tf.layers.batch_normalization(conv2d(h_pool2_t, W_conv3_t) + b_conv3_t, training = bn_training, name='target_bn3')
# 	h_conv3_t = lrelu(h_bn3_t)
# 	h_pool3_t = max_pool_2x2(h_conv3_t)
# 
# 	h_pool3_flat_t = tf.reshape(h_pool3_t, [-1, 14*14*32])
# 	W_fc1_t = weight_variable([14*14*32, 1], 'target_fc1_weight')
# 	b_fc1_t = bias_variable([1], 'target_fc1_bias')
# 	target_logit = tf.matmul(h_pool3_flat_t, W_fc1_t) + b_fc1_t

with tf.name_scope('target_model'):
	x_t = tf.placeholder("float", shape=[None, 109,109, 1])
	y_t = tf.placeholder("float", shape=[None, 1])
	W_conv1_t = weight_variable([5, 5, 1, 32], 'target_conv1_weight')
	b_conv1_t = bias_variable([32], 'target_conv1_bias')
	h_bn1_t = tf.layers.batch_normalization(conv2d(x_t, W_conv1_t)+b_conv1_t, training = bn_training, name='target_bn1')
	h_conv1_t = lrelu(h_bn1_t)
# 	h_pool1_t = max_pool_2x2(h_conv1_t)

	W_conv2_t = weight_variable([5, 5, 32, 32], 'target_conv2_weight')
	b_conv2_t = weight_variable([32], 'target_conv2_bias')
	h_bn2_t = tf.layers.batch_normalization(conv2d(h_conv1_t, W_conv2_t) + b_conv2_t, training = bn_training, name='target_bn2')
	h_conv2_t = lrelu(h_bn2_t)
	h_pool2_t = max_pool_2x2(h_conv2_t)

	W_conv3_t = weight_variable([5, 5, 32, 32], 'target_conv3_weight')
	b_conv3_t = weight_variable([32], 'target_conv3_bias')
	h_bn3_t = tf.layers.batch_normalization(conv2d(h_pool2_t, W_conv3_t) + b_conv3_t, training = bn_training, name='target_bn3')
	h_conv3_t = lrelu(h_bn3_t)
# 	h_pool3_t = max_pool_2x2(h_conv3_t)

	W_conv4_t = weight_variable([5, 5, 32, 32], 'target_conv4_weight')
	b_conv4_t = weight_variable([32], 'target_conv4_bias')
	h_bn4_t = tf.layers.batch_normalization(conv2d(h_conv3_t, W_conv4_t) + b_conv4_t, training = bn_training, name='target_bn4')
	h_conv4_t = lrelu(h_bn4_t)
	h_pool4_t = max_pool_2x2(h_conv4_t)

	h_pool4_flat_t = tf.reshape(h_pool4_t, [-1, 28*28*32])
	W_fc1_t = weight_variable([28*28*32, 1], 'target_fc1_weight')
	b_fc1_t = bias_variable([1], 'target_fc1_bias')
	target_logit = tf.matmul(h_pool4_flat_t, W_fc1_t) + b_fc1_t

target_vars_list = [v for v in tf.trainable_variables() if 'target_' in v.name]
saved_keys_list= ['base_conv1_weight', 'base_conv1_bias', 
				  'base_bn1_gamma', 'base_bn1_beta',
				  'base_conv2_weight', 'base_conv2_bias',
				  'base_bn2_gamma', 'base_bn2_beta', 
				  'base_conv3_weight', 'base_conv3_bias',
				  'base_bn3_gamma', 'base_bn3_beta',
				  'base_conv4_weight', 'base_conv4_bias',
				  'base_bn4_gamma', 'base_bn4_beta',
				  'base_fc1_weight', 'base_fc1_bias']
target_direct = {}
for key, var in zip(saved_keys_list, target_vars_list):
	target_direct[key] = var

target_saver = tf.train.Saver(target_direct)

with tf.Session() as sess:
	print('Testing source model loading')
	sess.run(tf.global_variables_initializer())
	source_saver.restore(sess, base_model_folder+'/base_model')
	test_source_logit = source_logit.eval(session=sess,feed_dict={x_s:Xs_tst})
	test_source_stat = np.exp(test_source_logit)
	test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
	test_target_logit = source_logit.eval(session=sess,feed_dict={x_s:Xt_tst})
	test_target_stat = np.exp(test_target_logit)
	test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
	print('>>>>>> Check the Initial Source Model Loading <<<<<<')
	print_red('Source to Source:{0:.4f} '.format(test_source_AUC))
	print_red('Source to Target:{0:.4f}'.format(test_target_AUC))

with tf.Session() as sess:
	print('Testing target model loading')
	sess.run(tf.global_variables_initializer())
	target_saver.restore(sess, base_model_folder+'/base_model')
	test_source_logit = target_logit.eval(session=sess,feed_dict={x_t:Xs_tst})
	test_source_stat = np.exp(test_source_logit)
	test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
	test_target_logit = target_logit.eval(session=sess,feed_dict={x_t:Xt_tst})
	test_target_stat = np.exp(test_target_logit)
	test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
	print('>>>>>> Check the Initial Source Model Loading <<<<<<')
	print_yellow('Source to Source:{0:.4f} '.format(test_source_AUC))
	print_yellow('Source to Target:{0:.4f}'.format(test_target_AUC))


def discriminator(X, hsize=[128, 32],reuse=False):
	with tf.variable_scope("discriminator",reuse=reuse):
		h1 = tf.layers.dense(X, hsize[0],activation=tf.nn.leaky_relu)
		h2 = tf.layers.dense(h1, hsize[1],activation=tf.nn.leaky_relu)
		out = tf.layers.dense(h2,1)
	return out

src_logits = discriminator(source_logit)
trg_logits = discriminator(target_logit, reuse = True)
# src_logits = discriminator(h_pool3_flat_s)
# trg_logits = discriminator(h_pool3_flat_t, reuse = True)
# src_logits = discriminator(h_pool4_flat_s)
# trg_logits = discriminator(h_pool4_flat_t, reuse = True)

disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits,labels=tf.ones_like(src_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits, labels=tf.zeros_like(trg_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits,labels=tf.ones_like(trg_logits)))

discr_vars_list = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

disc_step = tf.train.AdamOptimizer(d_lr).minimize(disc_loss, var_list=discr_vars_list)
gen_step = tf.train.AdamOptimizer(g_lr).minimize(gen_loss, var_list=target_vars_list)

D_loss_list = []
G_loss_list = []
test_auc_list = []
val_auc_list = []
# sess = tf.Session()
# d_lr = 1e-5
# g_lr = 1e-5
# nb_steps = 100000
# nd_steps = 10
# ng_steps = 10
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	source_saver.restore(sess, base_model_folder+'/base_model')
	target_saver.restore(sess, base_model_folder+'/base_model')
	for iteration in range(nb_steps):
		indices_s = np.random.randint(0, Xs_trn.shape[0], batch_size)
		batch_s = Xs_trn[indices_s,:]
		indices_t = np.random.randint(0, Xt_trn.shape[0], batch_size)
		batch_t = Xt_trn[indices_t,:]
		for _ in range(nd_steps):
			_, D_loss = sess.run([disc_step, disc_loss], feed_dict={x_s: batch_s, x_t: batch_t})
		for _ in range(ng_steps):
			_, G_loss = sess.run([gen_step, gen_loss], feed_dict={x_t: batch_t})
		## testing
	# 	if iterion%100:
		test_source_logit = source_logit.eval(session=sess,feed_dict={x_s:Xs_tst})
		test_source_stat = np.exp(test_source_logit)
		test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
		test_target_logit = target_logit.eval(session=sess,feed_dict={x_t:Xt_tst})
		test_target_stat = np.exp(test_target_logit)
		test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
		val_target_logit = target_logit.eval(session=sess,feed_dict={x_t:Xt_val})
		val_target_stat = np.exp(val_target_logit)
		val_target_AUC = roc_auc_score(yt_val, val_target_stat)
		# print results
		print_block(symbol = '-', nb_sybl = 60)
		print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}; S-test: {2:.4f}'.format(test_target_AUC, val_target_AUC, test_source_AUC))
		print_yellow('Loss: D {0:.4f}, G :{1:.4f}; Iter:{2}'.format(D_loss/2, G_loss, iteration))
		# save results
		D_loss_list.append(D_loss/2)
		G_loss_list.append(G_loss)
		test_auc_list.append(test_target_AUC)
		val_auc_list.append(val_target_AUC)
		print_yellow(os.path.basename(DA_model_folder))
		plot_loss(DA_model_folder, D_loss_list, G_loss_list, DA_model_folder+'/loss_{}.png'.format(DA_model_name))
		np.savetxt(os.path.join(DA_model_folder,'test_auc.txt'), test_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'val_auc.txt'), val_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'D_loss.txt'),D_loss_list)
		np.savetxt(os.path.join(DA_model_folder,'G_loss.txt'),G_loss_list)
		plot_auc_iterations(test_auc_list, val_auc_list, DA_model_folder+'/AUC_{}.png'.format(DA_model_name))
		# save models
		target_saver.save(sess, DA_model_folder +'/target')