import tensorflow as tf

import numpy as np
import os
import glob
from natsort import natsorted
from termcolor import colored 
import argparse
from sklearn.metrics import roc_auc_score
import scipy.io
import time

from load_data import *
from model import *

from functools import partial

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost

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
parser.add_argument("--lr", type = float)
parser.add_argument("--iters", type = int)
parser.add_argument("--bz", type = int)
parser.add_argument("--mmd_param", type = float)


args = parser.parse_args()
gpu_num = args.gpu
batch_size = args.bz
nb_steps = args.iters
mmd_param = args.mmd_param
lr = args.lr

if False:
    gpu_num = 6
    lr = 1e-5
    batch_size = 400
    nb_steps = 1000
    mmd_param = 10

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
# hyper-parameters
noise = 2.0
sig_rate = 0.035
source_model_name = 'cnn-4-bn-True-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-4.0k'
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
DA_model_name = 'mmd-{0:}-lr-{1:}-bz-{2:}-iter-{3:}'.format(mmd_param, lr, batch_size, nb_steps)
DA_model_folder = os.path.join(base_model_folder, DA_model_name)
generate_folder(DA_model_folder)
os.system('cp -f {} {}'.format(source_model_file+'*', DA_model_folder))

if source_model_name.split('-')[0] == 'cnn':
	nb_cnn = int(source_model_name.split('-')[1])
else:
	nb_cnn = 4

if source_model_name.split('-')[2] == 'bn':
	bn = bool(source_model_name.split('-')[3])
else:
	bn = False

xs = tf.placeholder("float", shape=[None, 109,109, 1])
ys = tf.placeholder("float", shape=[None, 1])
xt = tf.placeholder("float", shape=[None, 109,109, 1])
yt = tf.placeholder("float", shape=[None, 1])

conv_net_src, h_src, source_logit = conv_classifier(xs, nb_cnn = nb_cnn, fc_layers = [128,1],  bn = bn, scope_name = 'source')
conv_net_trg, h_trg, target_logit = conv_classifier(xt, nb_cnn = nb_cnn, fc_layers = [128,1],  bn = bn, scope_name = 'target')

source_vars_list = tf.trainable_variables('source')
source_key_list = [v.name[:-2].replace('source', 'base') for v in tf.trainable_variables('source')]
source_key_direct = {}
for key, var in zip(source_key_list, source_vars_list):
	source_key_direct[key] = var
source_saver = tf.train.Saver(source_key_direct, max_to_keep=nb_steps)

target_vars_list = tf.trainable_variables('target')
target_key_list = [v.name[:-2].replace('target', 'base') for v in tf.trainable_variables('target')]
target_key_direct = {}
for key, var in zip(target_key_list, target_vars_list):
	target_key_direct[key] = var
target_saver = tf.train.Saver(target_key_direct, max_to_keep=nb_steps)

with tf.variable_scope('mmd'):
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
    loss_value = maximum_mean_discrepancy(h_src, h_trg, kernel=gaussian_kernel)
    mmd_loss = mmd_param*tf.maximum(1e-4, loss_value)

gen_step = tf.train.AdamOptimizer(lr).minimize(mmd_loss, var_list=target_vars_list)

D_loss_list = []
test_auc_list = []
val_auc_list = []

## model loading verification
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	source_saver.restore(sess, source_model_file)
	target_saver.restore(sess, source_model_file)
	# source to source (target loading)
	print_yellow('>>>>>> Check the Initial Source Model Loading <<<<<<')
	test_source_logit_source = source_logit.eval(session=sess,feed_dict={xs:Xs_tst})
	test_source_stat_source = np.exp(test_source_logit_source)
	test_source_AUC_source = roc_auc_score(ys_tst, test_source_stat_source)
	print_yellow('Source loading: source-source:{0:.4f} '.format(test_source_AUC_source))
	# source to source (target loading)
	test_source_logit = target_logit.eval(session=sess,feed_dict={xt:Xs_tst})
	test_source_stat = np.exp(test_source_logit)
	test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
	# source to target (target loading)
	test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
	test_target_stat = np.exp(test_target_logit)
	test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
	print_yellow('Target loading: source-source:{0:.4f} source-target {1:.4f}'.format(test_source_AUC, test_target_AUC))

# nd_step_used = nd_steps
# ng_step_used = ng_steps
sess = tf.Session()
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	source_saver.restore(sess, source_model_file)
	target_saver.restore(sess, source_model_file)
	for iteration in range(nb_steps):
		indices_s = np.random.randint(0, Xs_trn.shape[0], batch_size)
		batch_s = Xs_trn[indices_s,:]
		indices_t = np.random.randint(0, Xt_trn.shape[0], batch_size)
		batch_t = Xt_trn[indices_t,:]
		_, D_loss = sess.run([gen_step, mmd_loss], feed_dict={xs: batch_s, xt: batch_t})	
		#testing
		test_source_logit = source_logit.eval(session=sess,feed_dict={xs:Xs_tst})
		test_source_stat = np.exp(test_source_logit)
		test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
		test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
		test_target_stat = np.exp(test_target_logit)
		test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
		val_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_val})
		val_target_stat = np.exp(val_target_logit)
		val_target_AUC = roc_auc_score(yt_val, val_target_stat)
		# print results
		print_block(symbol = '-', nb_sybl = 60)
		print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}; S-test: {2:.4f}'.format(test_target_AUC, val_target_AUC, test_source_AUC))
		print_yellow('MMD loss :{0:.4f}, Iter:{1:}'.format(D_loss, iteration))
		# save results
		D_loss_list.append(D_loss)
		test_auc_list.append(test_target_AUC)
		val_auc_list.append(val_target_AUC)
		print_yellow(os.path.basename(DA_model_folder))
		plot_loss(DA_model_folder, D_loss_list, D_loss_list, DA_model_folder+'/loss_{}.png'.format(DA_model_name))
		np.savetxt(os.path.join(DA_model_folder,'test_auc.txt'), test_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'val_auc.txt'), val_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'MMD_loss.txt'),D_loss_list)
		plot_auc_iterations(test_auc_list, val_auc_list, DA_model_folder+'/AUC_{}.png'.format(DA_model_name))
		# save models
		target_saver.save(sess, DA_model_folder +'/target')
