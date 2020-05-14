import tensorflow as tf
import tensorflow.math as tm

import numpy as np
import os
import glob
from natsort import natsorted
from termcolor import colored 
import argparse
from sklearn.metrics import roc_auc_score
import scipy.io
import time
from functools import partial

# user-defined tools
from load_data import load_Lumpy 
from model import conv_classifier

def str2bool(value):
    return value.lower() == 'true'

def plot_LOSS(file_name, train_loss_list, val_loss_list, test_loss_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	ax = fig.add_subplot(111)
	ax.plot(train_loss_list)
	ax.plot(val_loss_list)
	ax.plot(test_loss_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Loss')
	ax.legend(['D','S','T'])
	ax.set_xlim([0,len(train_loss_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_AUC(file_name, train_list, val_list, test_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
	ax.plot(train_list)
	ax.plot(val_list)
	ax.plot(test_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC')
	ax.legend(['Train','Valid','Test'])
	ax.set_xlim([0,len(train_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.system('mkdir -p {}'.format(folder))

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

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# plot and save the file
def plot_loss(file_name, loss, val_loss):
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
		ax.set_xlabel('Iterations/100')
		ax.legend(['source-loss', 'mmd-loss'], loc='upper left')  
		canvas = FigureCanvasAgg(fig)
		canvas.print_figure(f_out, dpi=80)

def plot_auc(target_file_name, val_auc_list, target_auc_list):
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

def plot_hist(file_name, x, y):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	kwargs = dict(alpha=0.6, bins=100, density= False, stacked=True)
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
	ax.hist(x, **kwargs, color='g', label='SA')
	ax.hist(y, **kwargs, color='r', label='Anomaly')
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('statistics')
	ax.set_ylabel('Frequency')
	ax.legend(['SA', 'SP'])
	ax.set_xlim([np.min(np.concatenate([x,y])), np.max(np.concatenate([x,y]))])
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
parser.add_argument("--gpu", type=int, default = 2)
parser.add_argument("--docker", type = str2bool, default = True)
parser.add_argument("--shared", type = str2bool, default = True)
parser.add_argument("--lr", type = float, default = 1e-5)
parser.add_argument("--iters", type = int, default = 10000)
parser.add_argument("--bz", type = int, default = 300)
parser.add_argument("--mmd_param", type = float, default = 1.0)
parser.add_argument("--trg_clf_param", type = float, default = 1.0)
parser.add_argument("--src_clf_param", type = float, default = 1.0)
parser.add_argument("--scratch", type = str2bool, default = True)
parser.add_argument("--nb_source", type = int, default = 100000)
parser.add_argument("--nb_target", type = int, default = 100000)
parser.add_argument("--nb_trg_labels", type = int, default = 0)
parser.add_argument("--fc_layer", type = int, default = 128)
parser.add_argument("--bn", type = str2bool, default = False)
parser.add_argument("--s_h", type = float, default = 40)
parser.add_argument("--s_blur", type = float, default = 0.5)
parser.add_argument("--s_noise", type = float, default = 10)
parser.add_argument("--t_h", type = float, default = 50)
parser.add_argument("--t_blur", type = float, default = 4.0)
parser.add_argument("--t_noise", type = float, default = 10)
# parser.add_argument("--clf_v", type = int, default = 1)
# parser.add_argument("--dataset", type = str, default = 'total')
parser.add_argument("--valid", type = int, default = 100)
parser.add_argument("--test", type = int, default = 200)

args = parser.parse_args()
print(args)

gpu = args.gpu
docker = args.docker
shared = args.shared
batch_size = args.bz
nb_steps = args.iters
mmd_param = args.mmd_param
lr = args.lr
nb_trg_labels = args.nb_trg_labels
source_scratch = args.scratch
fc_layer = args.fc_layer
bn = args.bn
trg_clf_param = args.trg_clf_param
src_clf_param = args.src_clf_param
# clf_v = args.clf_v
# dataset = args.dataset
s_h =args.s_h
s_blur = args.s_blur
s_noise = args.s_noise
t_h = args.t_h
t_blur = args.t_blur
t_noise = args.t_noise
valid = args.valid
test = args.test

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

if docker:
	output_folder ='/data/results/'
else:
	output_folder = 'data/'
print(output_folder)
source_folder = os.path.join(output_folder,'Lumpy/source')
target_folder = os.path.join(output_folder,'Lumpy/target')
source_model_name = 'cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k'
source_model_file = os.path.join(source_folder, source_model_name, 'source-best')
DA_folder = os.path.join(output_folder, 'Lumpy-Lumpy', source_model_name)
# load source data
nb_source = 100000
nb_target = 100000
Xs_trn, Xs_val, Xs_tst, ys_trn, ys_val, ys_tst = load_Lumpy(docker = docker, train = nb_source, valid = valid, test = test, height = s_h, blur= s_blur, noise = s_noise)
Xs_trn, Xs_val, Xs_tst = np.expand_dims(Xs_trn, axis = 3), np.expand_dims(Xs_val, axis = 3), np.expand_dims(Xs_tst, axis = 3)
ys_trn, ys_tst = ys_trn.reshape(-1,1), ys_tst.reshape(-1,1)
Xt_trn, Xt_val, Xt_tst, yt_trn, yt_val, yt_tst = load_Lumpy(docker = docker, train = nb_target, valid = valid, test = test, height = t_h, blur= t_blur, noise = t_noise)
Xt_trn, Xt_val, Xt_tst = np.expand_dims(Xt_trn, axis = 3), np.expand_dims(Xt_val, axis = 3), np.expand_dims(Xt_tst, axis = 3)
yt_trn, yt_val, yt_tst = yt_trn.reshape(-1,1), yt_val.reshape(-1,1), yt_tst.reshape(-1,1)
Xt_trn_l = np.concatenate([Xt_trn[0:nb_trg_labels,:],Xt_trn[nb_target:nb_target+nb_trg_labels,:]], axis = 0)
yt_trn_l = np.concatenate([yt_trn[0:nb_trg_labels,:],yt_trn[nb_target:nb_target+nb_trg_labels,:]], axis = 0)

# DA = os.path.join(output_folder, '{}-{}'.format(os.path.basename(source), os.path.basename(target)))
# generate_folder(DA)
# base_model_folder = os.path.join(DA, source_model_name)
# generate_folder(base_model_folder)
# copy the source weight file to the DA_model_folder
DA_model_name = 'Lumpy-mmd-{}-sclf{}-tclf-{}-lr-{}-bz-{}-scr-{}-shared-{}-bn-{}-labels-{}-val-{}-S-{}-{}-{}-T-{}-{}-{}-itrs-{}'\
				.format(mmd_param,src_clf_param,trg_clf_param,lr,batch_size,source_scratch,shared,bn, nb_trg_labels, valid, s_h, s_blur, s_noise, t_h, t_blur, t_noise, nb_steps)
DA_model_folder = os.path.join(DA_folder, DA_model_name)
generate_folder(DA_model_folder)
os.system('cp -f {} {}'.format(source_model_file+'*', DA_model_folder))

source_splits = source_model_name.split('-')
nb_cnn = 4
# bn = False
for i in range(len(source_splits)):
	if source_splits[i] == 'cnn':
		nb_cnn = int(source_splits[i+1])
# 	if source_splits[i] == 'bn':
# 		bn = str2bool(source_splits[i])

nb_cnn = 4
# bn = False
img_size = 64
xs = tf.placeholder("float", shape=[None, img_size, img_size, 1])
ys = tf.placeholder("float", shape=[None, 1])
xt = tf.placeholder("float", shape=[None, img_size, img_size, 1])
yt = tf.placeholder("float", shape=[None, 1])
xt1 = tf.placeholder("float", shape=[None, img_size, img_size, 1])   # input target image with labels
yt1 = tf.placeholder("float", shape=[None, 1])			  # input target image labels

if shared:
	target_scope = 'source'
	target_reuse = True
else:
	target_scope = 'target'
	target_reuse = False

conv_net_src, h_src, source_logit = conv_classifier(xs, nb_cnn = nb_cnn, fc_layers = [fc_layer,1],  bn = bn, scope_name = 'source')
conv_net_trg, h_trg, target_logit = conv_classifier(xt, nb_cnn = nb_cnn, fc_layers = [fc_layer,1],  bn = bn, scope_name = target_scope, reuse = target_reuse)
_, _, target_logit_l = conv_classifier(xt1, nb_cnn = nb_cnn, fc_layers = [fc_layer,1],  bn = bn, scope_name = target_scope, reuse = True)


source_vars_list = tf.trainable_variables('source')
source_key_list = [v.name[:-2].replace('source', 'base') for v in tf.trainable_variables('source')]
source_key_direct = {}
for key, var in zip(source_key_list, source_vars_list):
	source_key_direct[key] = var
source_key_direct_except_last_layer = {}
for key, var in zip(source_key_list[:-2], source_vars_list[:-2]):
	source_key_direct_except_last_layer[key] = var
source_saver = tf.train.Saver(source_key_direct, max_to_keep=nb_steps)
pre_trained_saver = tf.train.Saver(source_key_direct_except_last_layer, max_to_keep = nb_steps)

target_vars_list = tf.trainable_variables(target_scope)
target_key_list = [v.name[:-2].replace(target_scope, 'base') for v in tf.trainable_variables(target_scope)]
target_key_direct = {}
for key, var in zip(target_key_list, target_vars_list):
	target_key_direct[key] = var
target_saver = tf.train.Saver(target_key_direct, max_to_keep=nb_steps)
print(target_vars_list)

# source loss
src_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ys, logits = source_logit))
source_loss = src_clf_param*src_clf_loss
source_trn_ops = tf.train.AdamOptimizer(lr).minimize(source_loss, var_list = target_vars_list)

# mmd loss
sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
loss_value = maximum_mean_discrepancy(h_src, h_trg, kernel=gaussian_kernel)
mmd_loss = mmd_param*loss_value
mmd_trn_ops = tf.train.AdamOptimizer(lr).minimize(mmd_loss, var_list = target_vars_list)

if nb_trg_labels > 0:
	trg_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = yt1, logits = target_logit_l))
	target_loss = trg_clf_param*trg_clf_loss
	target_trn_ops = tf.train.AdamOptimizer(lr).minimize(target_loss, var_list = target_vars_list)

trg_loss_list, src_loss_lis, mmd_loss_list, trg_trn_auc_list, src_tst_auc_list, trg_val_auc_list, trg_tst_auc_list =\
	[], [], [], [], [], [], []
best_val_auc = -np.inf
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	for iteration in range(nb_steps):
		indices = np.random.randint(0, Xs_trn.shape[0]-1, batch_size)
		# train the source
		source_x = Xs_trn[indices,:];source_y = ys_trn[indices,:];sess.run(source_trn_ops, feed_dict={xs:source_x, ys: source_y})
		# train the feature extractors
		target_x = Xt_trn[indices,:];sess.run(mmd_trn_ops, feed_dict={xs:source_x, xt:target_x})
		# train the target
		if nb_trg_labels > 0:
			l_indices = np.random.randint(0, Xt_trn_l.shape[0]-1, 50)
			batch_x = Xt_trn_l[indices,:]; batch_y = yt_trn_l[indices,:]; sess.run(target_trn_ops, feed_dict={xt1:batch_x, yt1:batch_y})
		if iteration%100 == 0:
			src_loss = source_loss.eval(session=sess, feed_dict={xs:source_x, ys: source_y})
			MMD_loss = mmd_loss.eval(session=sess, feed_dict={xs:source_x, xt:target_x})
			if nb_trg_labels > 0:
				trg_loss = target_loss.eval(session=sess, feed_dict={xt1:batch_x, yt1:batch_y})
				trg_trn_stat = target_logit_l.eval(session=sess, feed_dict={xt1:batch_x})
				trg_trn_auc = roc_auc_score(batch_y, trg_trn_stat)
				print_yellow('Train with target labels:loss {0:.4f} auc {1:.4f}'.format(trg_loss, trg_trn_auc))
				trg_loss_list, trg_trn_auc_list = np.append(trg_loss_lis, trg_loss), np.append(trg_trn_auc_list, trg_trn_auc)
				np.savetxt(DA_model_folder+'/target_train_loss.txt',trg_loss)
				np.savetxt(DA_model_folder+'/target_train_auc.txt',trg_trn_auc)
			src_trn_stat = source_logit.eval(session=sess, feed_dict={xs:source_x}); src_auc = roc_auc_score(source_y, src_trn_stat)
			src_stat = source_logit.eval(session=sess, feed_dict={xs:Xs_tst}); src_auc = roc_auc_score(ys_tst, src_stat)
			trg_val_stat = target_logit.eval(session=sess, feed_dict={xt:Xt_val}); trg_val_auc = roc_auc_score(yt_val, trg_val_stat)
			trg_stat = target_logit.eval(session=sess, feed_dict={xt:Xt_tst}); trg_auc = roc_auc_score(yt_tst, trg_stat)
			src_loss_list, src_tst_auc_list = np.append(src_loss_lis, src_loss), np.append(src_tst_auc_list, src_auc)
			mmd_loss_list, trg_val_auc_list, trg_tst_auc_list = np.append(mmd_loss_list, MMD_loss), np.append(trg_val_auc_list, trg_val_auc), np.append(trg_tst_auc_list, trg_auc)
			np.savetxt(DA_model_folder+'/source_train_loss.txt', src_loss_list);np.savetxt(DA_model_folder+'/source_test_auc.txt', src_tst_auc_list)
			np.savetxt(DA_model_folder+'/mmd_train_loss.txt', trg_loss_list);np.savetxt(DA_model_folder+'/target_test_auc.txt',trg_tst_auc_list)
			np.savetxt(DA_model_folder+'/target_val_auc.txt', trg_val_auc_list)
			print_green('LOSS: src-test {0:.4f} mmd {1:.4f}; AUC: T-val {2:.4f} T-test {3:.4f} S-train {4:.4f} S-test {5:.4f}-iter-{6:}'.format(src_loss, MMD_loss, trg_val_auc, trg_auc, src_trn_auc, src_auc, iteration))
			print(DA_model_name)
			if best_val_auc < trg_val_auc:
				best_val_auc = trg_val_auc
				np.savetxt(DA_model_folder+'/best_stat.txt')
				target_saver.save(sess, DA_model_folder +'/best')
# 		if iteration%100 == 0:
# 			loss_trn = cost.eval(session = sess, feed_dict = {x:batch_x})
# 			loss_val = cost.eval(session = sess, feed_dict = {x:X_SA_val})
# 			loss_norm = cost.eval(session = sess, feed_dict = {x:X_SA_tst})
# 			loss_anomaly = cost.eval(session = sess, feed_dict = {x:X_SP_tst})
# 			# reconstructed images
# 			Yn = y.eval(session = sess, feed_dict = {x: X_SA_tst}); Ya = y.eval(session = sess, feed_dict = {x: X_SP_tst})
# 			y_recon = np.concatenate([Yn, Ya], axis = 0)
# 			# reconstruction errors-based detection
# 			norm_err_map = err_map.eval(session = sess, feed_dict = {x: X_SA_tst}); anomaly_err_map = err_map.eval(session = sess, feed_dict = {x: X_SP_tst})
# 			recon_err_map = np.concatenate([norm_err_map, anomaly_err_map], axis = 0)
# 			recon_errs = np.apply_over_axes(np.mean, recon_err_map, [1,2,3]).flatten(); AE_auc = roc_auc_score(yt, recon_errs)
# 			# print out results
# 			print_block(symbol = '-', nb_sybl = 50)
# 			print(model_name)
# 			print_yellow('LOSS: T {0:.4f}, V {1:.4f}, Norm {2:.4f}, Anomaly {3:.4f}; AUC: AE {4:.4f}, M: {5:.4f}; iter {6:}'.\
# 					format(loss_trn, loss_val, loss_norm, loss_anomaly, AE_auc, MP_auc, iteration))
# 			# save model
# 			saver.save(sess, model_folder +'/model', global_step= iteration)
# 			# save results
# 			loss_trn_list, loss_val_list, loss_norm_list, loss_anomaly_list, auc_list =\
# 				np.append(loss_trn_list, loss_trn), np.append(loss_val_list, loss_val),\
# 					np.append(loss_norm_list, loss_norm), np.append(loss_anomaly_list, loss_anomaly), np.append(auc_list, AE_auc)
# 			np.savetxt(model_folder+'/train_loss.txt', loss_trn_list); np.savetxt(model_folder+'/val_loss.txt', loss_val_list)
# 			np.savetxt(model_folder+'/norm_loss.txt', loss_norm_list); np.savetxt(model_folder+'/anomaly_loss.txt',loss_anomaly_list)
# 			plot_LOSS(model_folder+'/loss-{}.png'.format(model_name), 0, loss_trn_list, loss_val_list, loss_norm_list, loss_anomaly_list)
# 			np.savetxt(model_folder+'/AE_auc.txt', auc_list); plot_AUC(model_folder+'/auc-{}.png'.format(model_name), auc_list)
# 
# 			if best_loss_val > loss_val:
# 				best_loss_val = loss_val
# 				saver.save(sess, model_folder +'/best'); print_red('update best:{}'.format(model_name))
# 				np.savetxt(model_folder+'/AE_stat.txt', recon_errs); np.savetxt(model_folder+'/best_auc.txt',[AE_auc, MP_auc])
# 				plot_hist(model_folder+'/hist-{}.png'.format(model_name), recon_errs[:int(len(recon_errs)/2)], recon_errs[int(len(recon_errs)/2):])
# 				save_recon_images(model_folder+'/recon-{}.png'.format(model_name), Xt, y_recon, recon_err_map, fig_size = [11,5])
# 
# D_loss_list = []
# sC_loss_list = []
# tC_loss_list = []
# test_auc_list = []
# val_auc_list = []
# train_auc_list = []
# best_val_auc = 0
# 
# ## model loading verification
# with tf.Session() as sess:
# 	tf.global_variables_initializer().run(session=sess)
# # 	pre_trained_saver.restore(sess, source_model_file)
# 	source_saver.restore(sess, source_model_file)
# 	if not shared:
# 		target_saver.restore(sess, source_model_file)
# 	# source to source (target loading)
# 	print_yellow('>>>>>> Check the Initial Source Model Loading <<<<<<')
# 	test_source_logit_source = source_logit.eval(session=sess,feed_dict={xs:Xs_tst})
# 	test_source_stat_source = np.exp(test_source_logit_source)
# 	test_source_AUC_source = roc_auc_score(ys_tst, test_source_stat_source)
# 	print_yellow('Source loading: source-source:{0:.4f} '.format(test_source_AUC_source))
# 	# source to source (target loading)
# 	test_source_logit = target_logit.eval(session=sess,feed_dict={xt:Xs_tst})
# 	test_source_stat = np.exp(test_source_logit)
# 	test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
# 	# source to target (target loading)
# 	test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
# 	test_target_stat = np.exp(test_target_logit)
# 	test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
# 	print_yellow('Target loading: source-source:{0:.4f} source-target {1:.4f}'.format(test_source_AUC, test_target_AUC))
# 
# # nd_step_used = nd_steps
# # ng_step_used = ng_steps
# # sess = tf.Session()
# with tf.Session() as sess:
# 	tf.global_variables_initializer().run(session=sess)
# 	if not source_scratch:
# # 		pre_trained_saver.restore(sess, source_model_file)
# 		target_saver.restore(sess, source_model_file)
# 		if not shared:
# 			target_saver.restore(sess, source_model_file)
# 	for iteration in range(nb_steps):
# 		indices_s = np.random.randint(0, Xs_trn.shape[0]-1, batch_size)
# 		batch_s = Xs_trn[indices_s,:]
# 		batch_ys = ys_trn[indices_s,:]
# 		indices_t = np.random.randint(0, Xt_trn.shape[0]-1, batch_size)
# 		batch_t = Xt_trn[indices_t,:]
# 		# training
# 		if nb_trg_labels > 0:
# 			indices_tl = np.random.randint(0, 2*nb_trg_labels-1, 100)
# 			batch_xt_l, batch_yt_l = Xt_trn_l[indices_tl, :], yt_trn_l[indices_tl, :]
# 			_, D_loss, sC_loss, tC_loss, trg_digit = sess.run([gen_step, mmd_loss, src_clf_loss, trg_clf_loss, target_logit_l], feed_dict={xs: batch_s, xt: batch_t, ys: batch_ys, xt1:batch_xt_l, yt1:batch_yt_l})
# 			train_target_stat = np.exp(trg_digit)
# 			train_target_AUC = roc_auc_score(batch_yt_l, train_target_stat)
# 			train_auc_list.append(train_target_AUC)
# 			tC_loss_list.append(tC_loss)
# 			np.savetxt(os.path.join(DA_model_folder,'train_auc.txt'), val_auc_list)
# 			np.savetxt(os.path.join(DA_model_folder,'trg_clf_loss.txt'),tC_loss_list)
# 		else:
# 			_, D_loss, sC_loss = sess.run([gen_step, mmd_loss, src_clf_loss], feed_dict={xs: batch_s, xt: batch_t, ys: batch_ys})
# 		#testing
# 		test_source_logit = source_logit.eval(session=sess,feed_dict={xs:Xs_tst})
# 		test_source_stat = np.exp(test_source_logit)
# 		test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
# 		test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
# 		test_target_stat = np.exp(test_target_logit)
# 		test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
# 		val_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_val})
# 		val_target_stat = np.exp(val_target_logit)
# 		val_target_AUC = roc_auc_score(yt_val, val_target_stat)
# 		test_auc_list.append(test_target_AUC)
# 		val_auc_list.append(val_target_AUC)
# 		D_loss_list.append(D_loss)
# 		sC_loss_list.append(sC_loss)
# 		# save results
# 		np.savetxt(os.path.join(DA_model_folder,'test_auc.txt'), test_auc_list)
# 		np.savetxt(os.path.join(DA_model_folder,'val_auc.txt'), val_auc_list)
# 		np.savetxt(os.path.join(DA_model_folder,'MMD_loss.txt'),D_loss_list)
# 		np.savetxt(os.path.join(DA_model_folder,'src_clf_loss.txt'),sC_loss_list)
# 		# print and plot results
# 		print_block(symbol = '-', nb_sybl = 60)
# 		print_yellow(os.path.basename(DA_model_folder))
# 		if nb_trg_labels > 0:
# 			print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}, T-train {2:.4f}; S-test: {3:.4f}'.format(test_target_AUC, val_target_AUC, train_target_AUC, test_source_AUC))
# 			print_yellow('Loss: MMD:{0:.4f}, S:{1:.4f}, t:{2:.4f}, Iter:{3:}'.format(D_loss, sC_loss, tC_loss, iteration))
# 			plot_LOSS(DA_model_folder+'/loss_{}.png'.format(DA_model_name), D_loss_list, sC_loss_list, tC_loss_list)
# 			plot_AUCs(DA_model_folder+'/AUC_{}.png'.format(DA_model_name), train_auc_list, val_auc_list, test_auc_list)
# 		else:
# 			print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}; S-test: {2:.4f}'.format(test_target_AUC, val_target_AUC, test_source_AUC))
# 			print_yellow('Loss: MMD:{0:.4f}, S:{1:.4f}, Iter:{2:}'.format(D_loss, sC_loss, iteration))
# 			plot_loss(DA_model_folder, D_loss_list, sC_loss_list, DA_model_folder+'/loss_{}.png'.format(DA_model_name))
# 			plot_auc_iterations(test_auc_list, val_auc_list, DA_model_folder+'/AUC_{}.png'.format(DA_model_name))
# 		# save models
# 		if iteration%100==0:
# 			target_saver.save(sess, DA_model_folder +'/target', global_step= iteration)
# 		if best_val_auc < val_target_AUC:
# 			best_val_auc = val_target_AUC
# 			target_saver.save(sess, DA_model_folder+'/target_best')
# 			np.savetxt(os.path.join(DA_model_folder,'test_stat.txt'), test_target_stat)
# 			np.savetxt(os.path.join(DA_model_folder,'test_best_auc.txt'), [test_target_AUC])
# 			print_red('Update best:'+DA_model_folder)