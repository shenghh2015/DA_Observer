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
import gc

from load_data import *
from models2 import *
from helper_function import print_red, print_green, print_yellow, print_block, generate_folder
from helper_function import plot_src_trg_auc_iterations, plot_auc_dom_acc_iterations, plot_auc_iterations, plot_gradients
# from helper_function import maximum_mean_discrepancy, gaussian_kernel_matrix, compute_pairwise_distances
from helper_function import plot_loss, plot_AUCs_DomACC, plot_src_trg_AUCs, plot_AUCs, plot_LOSS, plot_feature_dist, plot_feature_pair_dist

from tf_tool import fp32, lerp, lerp_clip
from functools import partial

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
parser.add_argument("--d_cnn", type = int, default = 0)
parser.add_argument("--g_cnn", type=int, default = 6)
parser.add_argument("--fc", type=int, default = 256)
parser.add_argument("--d_bn", type=str2bool, default = False)
parser.add_argument("--g_bn", type=str2bool, default = False)
parser.add_argument("--c_lr", type=float, default = 1e-5)
parser.add_argument("--g_lr", type=float, default = 1e-5)
parser.add_argument("--d_lr", type=float, default = 1e-5)
parser.add_argument("--bz", type = int, default = 100)
parser.add_argument("--itr", type = int, default = 10000)
parser.add_argument("--d_weight", type = float, default = 1.0)
parser.add_argument("--t_weight", type = float, default = 1.0)
parser.add_argument("--s_weight", type = float, default = 1.0)
parser.add_argument("--labels", type = int, default = 0)
parser.add_argument("--dataset", type = str, default = 'total')

args = parser.parse_args();print(args)
gpu = args.gpu; docker = args.docker
d_cnn=args.d_cnn; g_cnn = args.g_cnn; fc_layer = args.fc; d_bn = args.d_bn; g_bn = args.g_bn
batch_size = args.bz; nb_steps = args.itr
lr = args.c_lr;g_lr = args.g_lr;d_lr=args.d_lr
d_weight = args.d_weight;t_weight = args.t_weight;s_weight = args.s_weight; 
nb_trg_labels = args.labels;dataset = args.dataset

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
output_folder = '/data/results' if docker else './data'; print('Results saved in: '+output_folder)
DA_model_name = 'mmd-dcnn-{}-gcnn-{}-fc-{}-dbn-{}-gbn-{}-clr-{}-glr-{}-dlr-{}-dw-{}-tw-{}-sw-{}-D-{}-L-{}-bz-{}-itr-{}'.\
	format(d_cnn, g_cnn, fc_layer, d_bn, g_bn, lr, g_lr, d_lr, d_weight, t_weight, s_weight, dataset, nb_trg_labels, batch_size, nb_steps)

noise = 2.0; sig_rate = 0.035
source = os.path.join(output_folder,'CLB');target = os.path.join(output_folder,'FDA')
DA = os.path.join(output_folder,'{}-{}'.format(os.path.basename(source), os.path.basename(target)))
DA_model_folder = os.path.join(DA, DA_model_name)
generate_folder(DA_model_folder)

# load source data
nb_source = 40000
Xs_trn, Xs_val, Xs_tst, ys_trn, ys_val, ys_tst = load_source(train = nb_source, sig_rate = sig_rate)
Xs_trn, Xs_val, Xs_tst = np.random.RandomState(2).normal(Xs_trn, noise), np.random.RandomState(0).normal(Xs_val, noise), np.random.RandomState(1).normal(Xs_tst, noise)
Xs_trn, Xs_val, Xs_tst = (Xs_trn-np.min(Xs_trn))/(np.max(Xs_trn)-np.min(Xs_trn)), (Xs_val-np.min(Xs_val))/(np.max(Xs_val)-np.min(Xs_val)), (Xs_tst-np.min(Xs_tst))/(np.max(Xs_tst)-np.min(Xs_tst))
Xs_trn, Xs_val, Xs_tst = np.expand_dims(Xs_trn, axis = 3), np.expand_dims(Xs_val, axis = 3), np.expand_dims(Xs_tst, axis = 3)
ys_tst = ys_tst.reshape(-1,1);ys_trn = ys_trn.reshape(-1,1)

# load target data
nb_target = 40000 if dataset == 'total' else 7100
Xt_trn, Xt_val, Xt_tst, yt_trn, yt_val, yt_tst = load_target(dataset = dataset, train = nb_target, valid = 100)
Xt_trn, Xt_val, Xt_tst = (Xt_trn-np.min(Xt_trn))/(np.max(Xt_trn)-np.min(Xt_trn)), (Xt_val-np.min(Xt_val))/(np.max(Xt_val)-np.min(Xt_val)), (Xt_tst-np.min(Xt_tst))/(np.max(Xt_tst)-np.min(Xt_tst))
Xt_trn, Xt_val, Xt_tst = np.expand_dims(Xt_trn, axis = 3), np.expand_dims(Xt_val, axis = 3), np.expand_dims(Xt_tst, axis = 3)
yt_trn, yt_val, yt_tst = yt_trn.reshape(-1,1), yt_val.reshape(-1,1), yt_tst.reshape(-1,1)
Xt_trn_l = np.concatenate([Xt_trn[0:nb_trg_labels,:],Xt_trn[nb_target:nb_target+nb_trg_labels,:]], axis = 0)
yt_trn_l = np.concatenate([yt_trn[0:nb_trg_labels,:],yt_trn[nb_target:nb_target+nb_trg_labels,:]], axis = 0)

## inputs
with tf.name_scope('input'):
	xs = tf.placeholder("float", shape=[None, 109,109, 1])
	ys = tf.placeholder("float", shape=[None, 1])
	xt = tf.placeholder("float", shape=[None, 109,109, 1])
	yt = tf.placeholder("float", shape=[None, 1])
	g_training = tf.placeholder_with_default(False, (), 'g_training')
# 	d_training = tf.placeholder_with_default(False, (), 'd_training')

target_scope = 'source'
## feature extractor and source classifier
conv_net_src, h_src, source_logit = conv_classifier(xs, g_cnn,[args.fc,1],g_bn,'source', bn_training = g_training)
conv_net_trg, h_trg, target_logit = conv_classifier(xt, g_cnn,[args.fc,1],g_bn,'source', reuse = True, bn_training = g_training)
# discriminator input
if d_cnn > 0:
	d_src_input = conv_net_src; d_trg_input = conv_net_trg
# 	mixed_shape = [batch_size,1,1,1]; norm_reduce_shape=[1,2,3]
else:
	d_src_input = h_src; d_trg_input = h_trg
# 	mixed_shape = [batch_size,1]; norm_reduce_shape=[1]

# src_scores = discriminator(d_src_input, d_cnn, [fc_layer, 1], d_bn, reuse = False, drop = 0, bn_training = d_training)
# trg_scores = discriminator(d_trg_input, d_cnn, [fc_layer, 1], d_bn, reuse = True, drop = 0, bn_training = d_training)
# mix input
# mixing_factors = tf.random_uniform(mixed_shape, 0.0, 1.0, dtype=d_trg_input.dtype)
# 	mixed_input = lerp(d_src_input, d_trg_input, mixing_factors)
# mixed_input =  d_src_input + (d_trg_input-d_src_input) * mixing_factors
# mixed_scores = discriminator(mixed_input, d_cnn, [fc_layer, 1], d_bn, reuse = True, drop = 0, bn_training = d_training)

# source loss
src_clf_loss = s_weight*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ys, logits = source_logit))

# mmd loss
sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
mmd_loss = maximum_mean_discrepancy(d_src_input, d_trg_input, kernel=gaussian_kernel)

# total loss
total_loss = s_weight*src_clf_loss + d_weight*mmd_loss
# optimization ops
# d_ops = tf.train.AdamOptimizer(d_lr).minimize(dis_loss, var_list=tf.trainable_variables('discriminator'))
# g_ops = tf.train.AdamOptimizer(g_lr).minimize(gen_loss, var_list=tf.trainable_variables(target_scope))
c_ops = tf.train.AdamOptimizer(lr).minimize(total_loss, var_list=tf.trainable_variables('source'))

## source and target variables
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

def model_test(sess, source_logit, xs, Xs_tst):
	logit_list = []; minbatch_size = 100; batch_idx = 0; nb_sample=Xs_tst.shape[0]
	while(batch_idx*minbatch_size<nb_sample):
		batch = Xs_tst[batch_idx*minbatch_size:min((batch_idx+1)*minbatch_size,nb_sample),:]
		logit_scores=source_logit.eval(session=sess,feed_dict={xs:batch,g_training:False})
		logit_list.append(logit_scores);batch_idx = batch_idx+1
	return np.concatenate(logit_list)

# D_loss_list = []
# G_loss_list = []
M_loss_list = []
sC_loss_list = []
tC_loss_list = []
test_auc_list = []
val_auc_list = []
src_test_list =[]
best_val_auc = 0
# nb_steps = 10
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	for iteration in range(nb_steps):
		indices_s=np.random.randint(0,Xs_trn.shape[0]-1,batch_size); batch_s = Xs_trn[indices_s,:]; batch_ys = ys_trn[indices_s,:]
		indices_t=np.random.randint(0,Xt_trn.shape[0]-1,batch_size); batch_t = Xt_trn[indices_t,:]
		_, sC_loss, M_loss, ttl_loss = sess.run([c_ops, src_clf_loss, mmd_loss, total_loss], feed_dict={xs:batch_s,ys:batch_ys, xt:batch_t, g_training:True})
# 		_, D_loss = sess.run([d_ops, -dis_loss],feed_dict={xs:batch_s,xt:batch_t,g_training:False,d_training: True})
# 		_, G_loss = sess.run([g_ops, gen_loss],feed_dict={xs:batch_s,xt:batch_t,g_training: True,d_training: False})
		if nb_trg_labels > 0:
			indices_tl = np.random.randint(0, 2*nb_trg_labels-1, 100);batch_xt_l,batch_yt_l=Xt_trn_l[indices_tl,:],yt_trn_l[indices_tl, :]
			_, tC_loss, trg_digit = sess.run([c_ops, src_clf_loss, target_logit], feed_dict={xs:batch_xt_l,ys:batch_yt_l,g_training:True})
			train_target_stat = np.exp(trg_digit);train_target_AUC = roc_auc_score(batch_yt_l, train_target_stat)
		if iteration%100 == 0:
			# test statistics
			test_source_logit = model_test(sess, source_logit, xs, Xs_tst)
# 			test_source_logit=source_logit.eval(session=sess,feed_dict={xs:Xs_tst,g_training:False,d_training:False})
			test_source_stat=np.exp(test_source_logit);test_source_AUC=roc_auc_score(ys_tst,test_source_stat);src_test_list.append(test_source_AUC)
# 			test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst, g_training: False, d_training: False})
			test_target_logit = model_test(sess, target_logit, xt, Xt_tst)
			test_target_stat = np.exp(test_target_logit);test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
			val_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_val, g_training: False})
			val_target_stat = np.exp(val_target_logit);val_target_AUC = roc_auc_score(yt_val, val_target_stat)
			test_auc_list.append(test_target_AUC);val_auc_list.append(val_target_AUC)
			np.savetxt(DA_model_folder+'/test_auc.txt', test_auc_list);np.savetxt(DA_model_folder+'/val_auc.txt', val_auc_list)
			# loss
			M_loss_list.append(M_loss)# ;D_loss_list.append(D_loss);sC_loss_list.append(sC_loss)
			np.savetxt(DA_model_folder+'/MMD_loss.txt',M_loss_list);# np.savetxt(DA_model_folder+'/G_loss.txt',G_loss_list);
			np.savetxt(DA_model_folder+'/src_clf_loss.txt',sC_loss_list)
			# print and plot results
			print_block(symbol = '-', nb_sybl = 60);print_yellow(os.path.basename(DA_model_folder))
			if nb_trg_labels > 0:
				train_auc_list.append(train_target_AUC);tC_loss_list.append(tC_loss)
				np.savetxt(os.path.join(DA_model_folder,'trg_train_auc.txt'), train_auc_list)
				np.savetxt(os.path.join(DA_model_folder,'trg_clf_loss.txt'),tC_loss_list)
				print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}, T-train {2:.4f}, S-test: {3:.4f}'.format(test_target_AUC, val_target_AUC, train_target_AUC, test_source_AUC))
				print_yellow('Loss: D:{:.4f}, S:{:.4f}, T:{:.4f}, Iter:{:}'.format(M_loss, sC_loss, tC_loss, iteration))
				plot_LOSS(DA_model_folder+'/loss_{}.png'.format(DA_model_name), M_loss_list, sC_loss_list, tC_loss_list)
				plot_loss(DA_model_folder, M_loss_list, M_loss_list, DA_model_folder+'/MMD_loss_{}.png'.format(DA_model_name))
				plot_src_trg_AUCs(DA_model_folder+'/AUC_src_{}.png'.format(DA_model_name), train_auc_list, val_auc_list, test_auc_list, src_test_list)
				plot_AUCs(DA_model_folder+'/AUC_trg_{}.png'.format(DA_model_name), train_auc_list, val_auc_list, test_auc_list)
			else:
				print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}, S-test: {2:.4f}'.format(test_target_AUC, val_target_AUC, test_source_AUC))
				print_yellow('Loss: D:{:.4f}, S:{:.4f}, Iter:{:}'.format(M_loss, sC_loss, iteration))
				plot_loss(DA_model_folder, M_loss_list, sC_loss_list, DA_model_folder+'/loss_{}.png'.format(DA_model_name))
				plot_loss(DA_model_folder, M_loss_list, M_loss_list, DA_model_folder+'/MMD_lOSS_{}.png'.format(DA_model_name))
				plot_src_trg_auc_iterations(test_auc_list, val_auc_list, src_test_list, DA_model_folder+'/AUC_src_{}.png'.format(DA_model_name))
			plot_auc_iterations(test_auc_list, val_auc_list, DA_model_folder+'/AUC_Final_{}.png'.format(DA_model_name))
			if best_val_auc < val_target_AUC:
				best_val_auc = val_target_AUC
				target_saver.save(sess, DA_model_folder+'/target_best')
				np.savetxt(os.path.join(DA_model_folder,'test_stat.txt'), test_target_stat)
				np.savetxt(os.path.join(DA_model_folder,'test_best_auc.txt'), [test_target_AUC])
				print_red('Update best:'+DA_model_folder)
			if iteration%1000 == 0:
				indices=np.random.randint(0,Xs_tst.shape[0],100)
				source_feat = h_src.eval(session=sess,feed_dict={xs: Xs_tst[indices,],g_training:False}); target_feat = h_trg.eval(session=sess,feed_dict={xt:Xt_tst[indices,],g_training:False})
				plot_feature_pair_dist(DA_model_folder+'/feat_{}_iter_{}.png'.format(DA_model_name, iteration), np.squeeze(source_feat), np.squeeze(target_feat), ys_tst[indices], yt_tst[indices], ['source', 'target'])
		gc.collect()