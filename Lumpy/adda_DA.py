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
from models2 import conv_classifier, discriminator
from helper_function import generate_folder, print_green, print_red, print_yellow, print_block
from helper_function import plot_ADDA_LOSS, plot_AUC, plot_ADDA_loss, plot_auc, plot_hist

def str2bool(value):
    return value.lower() == 'true'

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

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default = 2)
parser.add_argument("--docker", type = str2bool, default = True)
parser.add_argument("--shared", type = str2bool, default = False)
parser.add_argument("--lr", type = float, default = 1e-5)
parser.add_argument("--g_lr", type = float, default = 1e-4)
parser.add_argument("--d_lr", type = float, default = 4e-4)
parser.add_argument("--iters", type = int, default = 10000)
parser.add_argument("--bz", type = int, default = 300)
parser.add_argument("--mmd_param", type = float, default = 1.0)
parser.add_argument("--trg_clf_param", type = float, default = 1.0)
parser.add_argument("--src_clf_param", type = float, default = 1.0)
parser.add_argument("--scratch", type = str2bool, default = False)
parser.add_argument("--nb_source", type = int, default = 100000)
parser.add_argument("--nb_target", type = int, default = 100000)
parser.add_argument("--nb_trg_labels", type = int, default = 0)
parser.add_argument("--fc_layer", type = int, default = 128)
parser.add_argument("--dis_cnn", type = int, default = 0)
parser.add_argument("--dis_fc", type = int, default = 512)
parser.add_argument("--dis_bn", type = str2bool, default = True)
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
d_lr = args.d_lr
g_lr = args.g_lr
nb_source = args.nb_source
nb_target = args.nb_target
nb_trg_labels = args.nb_trg_labels
source_scratch = args.scratch
fc_layer = args.fc_layer
dis_cnn = args.dis_cnn
dis_fc = args.dis_fc
dis_bn = args.dis_bn
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
source_model_name = 'Lumpy-source-cnn-4-fc-128-lr-0.0001-bz-300-bn-True-T_V_T-100k_100_200-40.0_0.5_10.0-itrs-200000-v2'
source_model_file = os.path.join(source_folder, source_model_name, 'best')
DA_folder = os.path.join(output_folder, 'Lumpy-Lumpy', source_model_name)
# load source data
# nb_source = 100000
# nb_target = 100000
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
DA_model_name = 'Lumpy-adda-{}-sclf{}-tclf-{}-lr-{}-{}-{}-bz-{}-scr-{}-shared-{}-dis_bn-{}-labels-{}-val-{}-S-{}-{}-{}-T-{}-{}-{}-itrs-{}-trn-{}-{}-dis_fc-{}'\
				.format(mmd_param,src_clf_param,trg_clf_param,lr, d_lr, g_lr, batch_size,source_scratch,shared, dis_bn, nb_trg_labels, valid, s_h, s_blur, s_noise, t_h, t_blur, t_noise, nb_steps, nb_source, nb_target, dis_fc)
DA_model_folder = os.path.join(DA_folder, DA_model_name)
generate_folder(DA_model_folder)
os.system('cp -f {} {}'.format(source_model_file+'*', DA_model_folder))

source_splits = source_model_name.split('-')
nb_cnn = 4
# bn = False
for i in range(len(source_splits)):
	if source_splits[i] == 'cnn':
		nb_cnn = int(source_splits[i+1])
	if source_splits[i] == 'bn':
		bn = str2bool(source_splits[i+1])

# nb_cnn = 4
# bn = False
img_size = 64
xs = tf.placeholder("float", shape=[None, img_size, img_size, 1])
ys = tf.placeholder("float", shape=[None, 1])
xt = tf.placeholder("float", shape=[None, img_size, img_size, 1])
yt = tf.placeholder("float", shape=[None, 1])
xt1 = tf.placeholder("float", shape=[None, img_size, img_size, 1])   # input target image with labels
yt1 = tf.placeholder("float", shape=[None, 1])			  # input target image labels
is_training = tf.placeholder_with_default(False, (), 'is_training')

if shared:
	target_scope = 'source'
	target_reuse = True
else:
	target_scope = 'target'
	target_reuse = False

conv_net_src, h_src, source_logit = conv_classifier(xs, nb_cnn = nb_cnn, fc_layers = [fc_layer,1],  bn = bn, scope_name = 'source', bn_training = is_training)
conv_net_trg, h_trg, target_logit = conv_classifier(xt, nb_cnn = nb_cnn, fc_layers = [fc_layer,1],  bn = bn, scope_name = target_scope, reuse = target_reuse, bn_training = is_training)
_, _, target_logit_l = conv_classifier(xt1, nb_cnn = nb_cnn, fc_layers = [fc_layer,1],  bn = bn, scope_name = target_scope, reuse = True, bn_training = is_training)

source_vars_list = tf.global_variables('source')
source_key_list = [v.name[:-2].replace('source', 'base') for v in tf.global_variables('source')]
source_key_direct = {}
for key, var in zip(source_key_list, source_vars_list):
	source_key_direct[key] = var
source_key_direct_except_last_layer = {}
for key, var in zip(source_key_list[:-2], source_vars_list[:-2]):
	source_key_direct_except_last_layer[key] = var
source_saver = tf.train.Saver(source_key_direct, max_to_keep=nb_steps)
pre_trained_saver = tf.train.Saver(source_key_direct_except_last_layer, max_to_keep = nb_steps)

target_vars_list = tf.global_variables(target_scope)
target_key_list = [v.name[:-2].replace(target_scope, 'base') for v in tf.global_variables(target_scope)]
target_key_direct = {}
for key, var in zip(target_key_list, target_vars_list):
	target_key_direct[key] = var
target_saver = tf.train.Saver(target_key_direct, max_to_keep=nb_steps)
print(target_vars_list)

# source loss
src_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ys, logits = source_logit))
source_loss = src_clf_param*src_clf_loss
source_trn_ops = tf.train.AdamOptimizer(lr).minimize(source_loss, var_list = tf.trainable_variables('source'))

# dis_cnn = 0
if dis_cnn > 0:
	src_logits = discriminator(conv_net_src, nb_cnn = dis_cnn, fc_layers = [128, 1], bn = dis_bn, bn_training = is_training)
	trg_logits = discriminator(conv_net_trg, nb_cnn = dis_cnn, fc_layers = [128, 1], bn = dis_bn, reuse = True, bn_training = is_training)
else:
# 	dis_bn = True
# 	dis_fc = 1024
	src_logits = discriminator(h_src, nb_cnn = 0, fc_layers = [dis_fc, 1], bn = dis_bn, bn_training = is_training)
	trg_logits = discriminator(h_trg, nb_cnn = 0, fc_layers = [dis_fc, 1], bn = dis_bn, reuse = True, bn_training = is_training)

disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits,labels=tf.ones_like(src_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits, labels=tf.zeros_like(trg_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits,labels=tf.zeros_like(src_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits, labels=tf.ones_like(trg_logits)))

# d_lr = 4e-4
# g_lr = 1e-4
disc_trn_ops = tf.train.AdamOptimizer(d_lr).minimize(disc_loss, var_list = tf.trainable_variables('discriminator'))
gen_trn_ops = tf.train.AdamOptimizer(g_lr).minimize(gen_loss, var_list = tf.trainable_variables(target_scope))
# mmd loss
# sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
# gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
# loss_value = maximum_mean_discrepancy(h_src, h_trg, kernel=gaussian_kernel)
# mmd_loss = mmd_param*loss_value
# mmd_trn_ops = tf.train.AdamOptimizer(lr).minimize(mmd_loss, var_list = target_vars_list)

if nb_trg_labels > 0:
	trg_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = yt1, logits = target_logit_l))
	target_loss = trg_clf_param*trg_clf_loss
	target_trn_ops = tf.train.AdamOptimizer(lr).minimize(target_loss, var_list = tf.trainable_variables(target_scope))

trg_loss_list, src_loss_list, d_loss_list, g_loss_list, trg_trn_auc_list, src_tst_auc_list, trg_val_auc_list, trg_tst_auc_list =\
	[], [], [], [], [], [], [], []
best_val_auc = -np.inf
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	if not source_scratch:
		source_saver.restore(sess, source_model_file); target_saver.restore(sess, source_model_file)
		src_stat = target_logit.eval(session=sess, feed_dict={xt:Xs_tst, is_training: False}); src_auc = roc_auc_score(ys_tst, src_stat)
		trg_stat = target_logit.eval(session=sess, feed_dict={xt:Xt_tst, is_training: False}); trg_auc = roc_auc_score(yt_tst, trg_stat)
		print('AUC: S-S {0:.4f} S-T {1:.4f}'.format(src_auc, trg_auc))
	for iteration in range(nb_steps):
		indices = np.random.randint(0, Xs_trn.shape[0]-1, batch_size)
		source_x = Xs_trn[indices,:];target_x = Xt_trn[indices,:]; source_y = ys_trn[indices,:]
		# train the source
		if shared:
			sess.run(source_trn_ops, feed_dict={xs:source_x, ys: source_y, is_training: True})
		# train the discriminator
		sess.run(disc_trn_ops, feed_dict={xs:source_x, xt:target_x, is_training: True})
		# train the feature extractors
		sess.run(gen_trn_ops, feed_dict={xs:source_x, xt:target_x, is_training: True})
		# train the target
		if nb_trg_labels > 0:
			l_indices = np.random.randint(0, Xt_trn_l.shape[0]-1, 50)
			batch_x = Xt_trn_l[l_indices,:]; batch_y = yt_trn_l[l_indices,:]; sess.run(target_trn_ops, feed_dict={xt1:batch_x, yt1:batch_y, is_training: True})
		if iteration%100 == 0:
			src_loss = source_loss.eval(session=sess, feed_dict={xs:source_x, ys: source_y, is_training: False})
			d_loss = disc_loss.eval(session=sess, feed_dict={xs:source_x, xt:target_x, is_training: False})
			g_loss = gen_loss.eval(session=sess, feed_dict={xs:source_x, xt:target_x, is_training: False})
			if nb_trg_labels > 0:
				trg_loss = target_loss.eval(session=sess, feed_dict={xt1:batch_x, yt1:batch_y, is_training: False})
				trg_trn_stat = target_logit_l.eval(session=sess, feed_dict={xt1:batch_x, is_training: False})
				trg_trn_auc = roc_auc_score(batch_y, trg_trn_stat)
				print_yellow('Train with target labels:loss {0:.4f} auc {1:.4f}'.format(trg_loss, trg_trn_auc))
				trg_loss_list, trg_trn_auc_list = np.append(trg_loss_list, trg_loss), np.append(trg_trn_auc_list, trg_trn_auc)
				np.savetxt(DA_model_folder+'/target_train_loss.txt',trg_loss_list)
				np.savetxt(DA_model_folder+'/target_train_auc.txt',trg_trn_auc_list)
			src_trn_stat = source_logit.eval(session=sess, feed_dict={xs:source_x, is_training: False}); src_trn_auc = roc_auc_score(source_y, src_trn_stat)
			src_stat = source_logit.eval(session=sess, feed_dict={xs:Xs_tst, is_training: False}); src_auc = roc_auc_score(ys_tst, src_stat)
			trg_val_stat = target_logit.eval(session=sess, feed_dict={xt:Xt_val, is_training: False}); trg_val_auc = roc_auc_score(yt_val, trg_val_stat)
			trg_stat = target_logit.eval(session=sess, feed_dict={xt:Xt_tst, is_training: False}); trg_auc = roc_auc_score(yt_tst, trg_stat)
			src_loss_list, src_tst_auc_list = np.append(src_loss_list, src_loss), np.append(src_tst_auc_list, src_auc)
			g_loss_list.append(g_loss)
			d_loss_list, trg_val_auc_list, trg_tst_auc_list = np.append(d_loss_list, d_loss), np.append(trg_val_auc_list, trg_val_auc), np.append(trg_tst_auc_list, trg_auc)
			np.savetxt(DA_model_folder+'/source_train_loss.txt', src_loss_list);np.savetxt(DA_model_folder+'/source_test_auc.txt', src_tst_auc_list)
			np.savetxt(DA_model_folder+'/gen_loss.txt', g_loss_list);np.savetxt(DA_model_folder+'/disc_loss.txt', d_loss_list);
			np.savetxt(DA_model_folder+'/DA_test_auc.txt',trg_tst_auc_list); np.savetxt(DA_model_folder+'/DA_val_auc.txt', trg_val_auc_list)
			print_green('LOSS: src-test {0:.4f} D {1:.4f} G {2:.4f}; AUC: T-val {3:.4f} T-test {4:.4f} S-train {5:.4f} S-test {6:.4f}-iter-{7:}'.format(src_loss, d_loss, g_loss, trg_val_auc, trg_auc, src_trn_auc, src_auc, iteration))
			print(DA_model_name)
			if nb_trg_labels > 0:
				plot_AUC(DA_model_folder + '/auc-full_{}.png'.format(DA_model_name), trg_trn_auc_list, trg_val_auc_list, trg_tst_auc_list)
				plot_ADDA_LOSS(DA_model_folder + '/loss-full_{}.png'.format(DA_model_name), trg_loss_list, src_loss_list, d_loss_list, g_loss_list)			
			plot_auc(DA_model_folder + '/auc_{}.png'.format(DA_model_name), trg_val_auc_list, trg_tst_auc_list)
			plot_ADDA_loss(DA_model_folder + '/loss_{}.png'.format(DA_model_name), src_loss_list, d_loss_list, g_loss_list)
			if best_val_auc < trg_val_auc:
				best_val_auc = trg_val_auc
				np.savetxt(DA_model_folder+'/best_stat.txt', trg_stat)
				target_saver.save(sess, DA_model_folder +'/best')
				plot_hist(DA_model_folder + '/hist_{}.png'.format(DA_model_name), trg_stat[:int(len(trg_stat)/2)], trg_stat[int(len(trg_stat)/2):])