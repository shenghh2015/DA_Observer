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

from load_data import *
# from model import *
from loss import *
from models2 import *
from helper_function import print_red, print_green, print_yellow, print_block, generate_folder
from helper_function import plot_src_trg_auc_iterations, plot_auc_dom_acc_iterations, plot_auc_iterations, plot_gradients
from helper_function import maximum_mean_discrepancy, gaussian_kernel_matrix, compute_pairwise_distances
from helper_function import plot_loss, plot_AUCs_DomACC, plot_src_trg_AUCs, plot_AUCs, plot_LOSS, plot_feature_dist, plot_feature_pair_dist

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
parser.add_argument("--gpu", type=int, default = 0)
parser.add_argument("--docker", type = str2bool, default = True)
parser.add_argument("--shared", type = str2bool, default = True)
parser.add_argument("--dis_cnn", type = int, default = 0)
parser.add_argument("--g_cnn", type=int, default = 4)
parser.add_argument("--g_lr", type=float, default = 1e-5)
parser.add_argument("--d_lr", type=float, default = 1e-5)
parser.add_argument("--lsmooth", type=str2bool, default = False)
parser.add_argument("--dis_fc", type=int, default = 128)
parser.add_argument("--dis_bn", type=str2bool, default = True)
parser.add_argument("--nD", type = int, default = 1)
parser.add_argument("--nG", type = int, default = 1)
parser.add_argument("--acc_up", type = float, default = 0.8)
parser.add_argument("--acc_down", type = float, default = 0.3)
parser.add_argument("--lr", type = float, default = 1e-5)
parser.add_argument("--iters", type = int, default = 1000)
parser.add_argument("--bz", type = int, default = 400)
parser.add_argument("--dis_param", type = float, default = 1.0)
parser.add_argument("--mmd_param", type = float, default = 1.0)
parser.add_argument("--trg_clf_param", type = float, default = 1.0)
parser.add_argument("--src_clf_param", type = float, default = 1.0)
parser.add_argument("--source_scratch", type = str2bool, default = True)
parser.add_argument("--nb_trg_labels", type = int, default = 0)
parser.add_argument("--fc_layer", type = int, default = 128)
parser.add_argument("--den_bn", type = str2bool, default = False)
parser.add_argument("--clf_v", type = int, default = 1)
parser.add_argument("--dataset", type = str, default = 'dense')
parser.add_argument("--drop", type = float, default = 0)

args = parser.parse_args()
print(args)

gpu_num = args.gpu
docker = args.docker
shared = args.shared
batch_size = args.bz
nb_steps = args.iters
dis_param = args.dis_param
mmd_param = args.mmd_param
dis_cnn = args.dis_cnn
dis_fc = args.dis_fc
dis_bn = args.dis_bn
nd_steps = args.nD
ng_steps = args.nG
acc_up = args.acc_up
acc_down = args.acc_down
lr = args.lr
nb_trg_labels = args.nb_trg_labels
source_scratch = args.source_scratch
fc_layer = args.fc_layer
den_bn = args.den_bn
dis_bn = args.dis_bn
trg_clf_param = args.trg_clf_param
src_clf_param = args.src_clf_param
clf_v = args.clf_v
dataset = args.dataset
g_lr = args.g_lr
d_lr = args.d_lr
lsmooth = args.lsmooth
drop = args.drop

g_cnn = args.g_cnn

if False:
	gpu_num = 1
	lr = 1e-5
	batch_size = 400
	nb_steps = 1000
	dis_param = 1.0
	nb_trg_labels = 0
	source_scratch = True
	docker = True
	shared = True
	fc_layer = 128
	den_bn = False

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

if docker:
	output_folder ='/data/results'
else:
	output_folder = 'data'

print(output_folder)
# hyper-parameters
noise = 2.0
sig_rate = 0.035
# source_model_name = 'cnn-4-bn-True-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-4.0k'
# source_model_name = 'cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k'
# source_model_name = 'cnn-6-bn-True-noise-2.0-trn-100000-sig-0.035-bz-300-lr-1e-05-Adam-stp-100.0k-clf_v1'
# source_model_name = 'cnn-6-bn-True-noise-2.0-trn-100000-sig-0.035-bz-300-lr-1e-05-Adam-stp-100.0k-clf_v1'
if den_bn:
	source_model_name = 'cnn-4-bn-True-noise-2.0-trn-100000-sig-0.035-bz-300-lr-1e-05-Adam-stp-100.0k-clf_v1'
else:
	source_model_name = 'cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-300-lr-1e-05-Adam-stp-100.0k-clf_v1'
# load source data
# source = '/data/results/CLB'
# target = '/data/results/FDA'
source = os.path.join(output_folder,'CLB')
target = os.path.join(output_folder,'FDA')
source_model_file = os.path.join(source, source_model_name, 'source-best')

# load source data
nb_source = 100000
Xs_trn, Xs_val, Xs_tst, ys_trn, ys_val, ys_tst = load_source(train = nb_source, sig_rate = sig_rate)
Xs_trn, Xs_val, Xs_tst = np.random.RandomState(2).normal(Xs_trn, noise), np.random.RandomState(0).normal(Xs_val, noise), np.random.RandomState(1).normal(Xs_tst, noise)
Xs_trn, Xs_val, Xs_tst = (Xs_trn-np.min(Xs_trn))/(np.max(Xs_trn)-np.min(Xs_trn)), (Xs_val-np.min(Xs_val))/(np.max(Xs_val)-np.min(Xs_val)), (Xs_tst-np.min(Xs_tst))/(np.max(Xs_tst)-np.min(Xs_tst))
Xs_trn, Xs_val, Xs_tst = np.expand_dims(Xs_trn, axis = 3), np.expand_dims(Xs_val, axis = 3), np.expand_dims(Xs_tst, axis = 3)
ys_tst = ys_tst.reshape(-1,1)
ys_trn = ys_trn.reshape(-1,1)
# load target data
if dataset == 'dense':
	nb_target = 7100
elif dataset == 'hetero':
	nb_target = 36000
elif dataset == 'scattered':
	nb_target = 33000
elif dataset == 'fatty':
	nb_target = 9000
elif dataset == 'total':
	nb_target = 85000
Xt_trn, Xt_val, Xt_tst, yt_trn, yt_val, yt_tst = load_target(dataset = dataset, train = nb_target, valid = 100)
Xt_trn, Xt_val, Xt_tst = (Xt_trn-np.min(Xt_trn))/(np.max(Xt_trn)-np.min(Xt_trn)), (Xt_val-np.min(Xt_val))/(np.max(Xt_val)-np.min(Xt_val)), (Xt_tst-np.min(Xt_tst))/(np.max(Xt_tst)-np.min(Xt_tst))
Xt_trn, Xt_val, Xt_tst = np.expand_dims(Xt_trn, axis = 3), np.expand_dims(Xt_val, axis = 3), np.expand_dims(Xt_tst, axis = 3)
yt_trn, yt_val, yt_tst = yt_trn.reshape(-1,1), yt_val.reshape(-1,1), yt_tst.reshape(-1,1)
Xt_trn_l = np.concatenate([Xt_trn[0:nb_trg_labels,:],Xt_trn[nb_target:nb_target+nb_trg_labels,:]], axis = 0)
yt_trn_l = np.concatenate([yt_trn[0:nb_trg_labels,:],yt_trn[nb_target:nb_target+nb_trg_labels,:]], axis = 0)
# DA = '/data/results/{}-{}'.format(os.path.basename(source), os.path.basename(target))
DA = os.path.join(output_folder, '{}-{}'.format(os.path.basename(source), os.path.basename(target)))
generate_folder(DA)
base_model_folder = os.path.join(DA, source_model_name)
generate_folder(base_model_folder)
# copy the source weight file to the DA_model_folder
DA_model_name = 'mmd_wd-{0:}-glr-{1:}-dlr-{2:}-bz-{3:}-iter-{4:}-scr-{5:}-shar-{6:}-dis_fc-{7:}-bn-{8:}-tclf-{9:}-sclf-{10:}-tlabels-{11:}-{12:}-cnn-{13:}-dis_bn-{14:}-gcnn-{15:}-smooth-{16:}-drop-{17:}-lr-{18:}-mmd-{19:}'.format(dis_param, g_lr,  d_lr, batch_size, nb_steps, source_scratch, shared, dis_fc, den_bn, trg_clf_param, src_clf_param, nb_trg_labels, dataset, dis_cnn, dis_bn, g_cnn, lsmooth, drop, lr, mmd_param)
DA_model_folder = os.path.join(base_model_folder, DA_model_name)
generate_folder(DA_model_folder)
os.system('cp -f {} {}'.format(source_model_file+'*', DA_model_folder))

if source_model_name.split('-')[0] == 'cnn':
	nb_cnn = int(source_model_name.split('-')[1])
else:
	nb_cnn = 4


xs = tf.placeholder("float", shape=[None, 109,109, 1])
ys = tf.placeholder("float", shape=[None, 1])
xt = tf.placeholder("float", shape=[None, 109,109, 1])
yt = tf.placeholder("float", shape=[None, 1])
xt1 = tf.placeholder("float", shape=[None, 109,109, 1])   # input target image with labels
yt1 = tf.placeholder("float", shape=[None, 1])			  # input target image labels
is_training = tf.placeholder_with_default(False, (), 'is_training')
dis_training = tf.placeholder_with_default(False, (), 'dis_training')

if shared:
	target_scope = 'source'
	target_reuse = True
else:
	target_scope = 'target'
	target_reuse = False

conv_net_src, h_src, source_logit = conv_classifier(xs, nb_cnn = g_cnn, fc_layers = [fc_layer,1],  bn = den_bn, scope_name = 'source', bn_training = is_training)
# flat1 = tf.layers.flatten(conv_net_src)
conv_net_trg, h_trg, target_logit = conv_classifier(xt, nb_cnn = g_cnn, fc_layers = [fc_layer,1],  bn = den_bn, scope_name = target_scope, reuse = target_reuse, bn_training = is_training)
_, _, target_logit_l = conv_classifier(xt1, nb_cnn = g_cnn, fc_layers = [fc_layer,1],  bn = den_bn, scope_name = target_scope, reuse = True, bn_training = is_training)
# flat2 = tf.layers.flatten(conv_net_trg)

source_vars_list = tf.global_variables('source')
# source_conv_list = tf.trainable_variables('source/conv')
# source_clf_list = tf.trainable_variables('source/classifier')
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
src_clf_loss = src_clf_param*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ys, logits = source_logit))

# mmd loss
sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
loss_value = maximum_mean_discrepancy(h_src, h_trg, kernel=gaussian_kernel)
mmd_loss = mmd_param*loss_value
mmd_trn_ops = tf.train.AdamOptimizer(lr).minimize(mmd_loss, var_list = tf.trainable_variables(target_scope))
# dis_cnn = 4
if dis_cnn > 0:
	src_logits = discriminator(conv_net_src, nb_cnn = dis_cnn, fc_layers = [dis_fc, 1], bn = dis_bn,  drop = drop, bn_training = dis_training)
	trg_logits = discriminator(conv_net_trg, nb_cnn = dis_cnn, fc_layers = [dis_fc, 1], bn = dis_bn, reuse = True, drop = drop, bn_training = dis_training)
else:
	src_logits = discriminator(h_src, nb_cnn = 0, fc_layers = [dis_fc, 1], bn = dis_bn, drop = drop, bn_training = dis_training)
	trg_logits = discriminator(h_trg, nb_cnn = 0, fc_layers = [dis_fc, 1], bn = dis_bn, reuse = True, drop = drop, bn_training = dis_training)

# if lsmooth:
# 	disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits,labels=tf.ones_like(src_logits)-0.1) + tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits, labels=tf.zeros_like(trg_logits)+0.1))
# else:
def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def lerp(a, b, t):
    with tf.name_scope('Lerp'):
        return a + (b - a) * t

def lerp_clip(a, b, t):
    with tf.name_scope('LerpClip'):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
# WD GAN loss
gen_loss = tf.reduce_mean(src_logits) - tf.reduce_mean(trg_logits)

wgan_lambda     = 10.0     # Weight for the gradient penalty term.
wgan_epsilon    = 0.001    # Weight for the epsilon term, \epsilon_{drift}.
wgan_target     = 1.0      # Target value for gradient magnitudes.
fakes, reals = conv_net_trg, conv_net_src; minibatch_size = batch_size
fake_scores_out = trg_logits 
real_scores_out = src_logits
disc_loss = tf.reduce_mean(fake_scores_out) - tf.reduce_mean(real_scores_out)

mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fakes.dtype)
mixed_input = lerp(reals, fakes, mixing_factors)
mixed_scores = discriminator(mixed_input, nb_cnn = dis_cnn, fc_layers = [dis_fc, 1], bn = dis_bn, reuse = True, drop = drop, bn_training = dis_training)
mixed_loss = tf.reduce_sum(mixed_scores)
mixed_grads = tf.gradients(mixed_loss, [mixed_input])[0]
mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
gradient_penalty = tf.reduce_mean(tf.square(mixed_norms - wgan_target))
disc_loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

epsilon_penalty = tf.reduce_mean(tf.square(real_scores_out))
disc_loss += epsilon_penalty * wgan_epsilon
#disc_loss = D_wgangp_acgan(reals = conv_net_src, fakes = conv_net_trg, minibatch_size = batch_size, dis_training = dis_training, dis_cnn = dis_cnn, fc_layers = [dis_fc, 1], dis_bn = dis_bn)

disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits,labels=tf.ones_like(src_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits, labels=tf.zeros_like(trg_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits,labels=tf.zeros_like(src_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits, labels=tf.ones_like(trg_logits)))

total_loss_no_labels = dis_param* gen_loss + src_clf_param*src_clf_loss
total_loss = total_loss_no_labels
discr_vars_list = tf.trainable_variables('discriminator')

# total_loss = mmd_loss + src_clf_param*src_clf_loss

if nb_trg_labels > 0:
	trg_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = yt1, logits = target_logit_l))
	total_loss = total_loss + trg_clf_param*trg_clf_loss
	trg_clf_step = tf.train.AdamOptimizer(lr).minimize(trg_clf_loss, var_list=tf.trainable_variables(target_scope))
disc_step = tf.train.AdamOptimizer(d_lr).minimize(disc_loss, var_list=discr_vars_list)
gen_step = tf.train.AdamOptimizer(g_lr).minimize(gen_loss, var_list=tf.trainable_variables(target_scope))
src_clf_step = tf.train.AdamOptimizer(lr).minimize(src_clf_loss, var_list=tf.trainable_variables('source'))
#gen_step_no_labels = tf.train.AdamOptimizer(g_lr).minimize(total_loss_no_labels, var_list=tf.trainable_variables(target_scope))

# tf.keras.backend.clear_session()
## compute the gradients
# dis_gradients = tf.gradients(disc_loss, discr_vars_list)
# dis_gradients = list(filter(None.__ne__, dis_gradients))
# gen_gradients = tf.gradients(gen_loss, tf.trainable_variables(target_scope)[:-2])
# gen_gradients = list(filter(None.__ne__, gen_gradients))

D_loss_list = []
G_loss_list = []
mmd_loss_list = []
sC_loss_list = []
tC_loss_list = []
test_auc_list = []
val_auc_list = []
dom_acc_list = []
train_auc_list = []
src_test_list =[]
best_val_auc = 0
train_target_AUC = 0.5
tC_loss = 1.4

# gradients
D_grad_list1, D_grad_list2, G_grad_list1, G_grad_list2 = [], [], [], []

nd_step_used = nd_steps
ng_step_used = ng_steps
# sess = tf.Session()
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	if not source_scratch:
# 		pre_trained_saver.restore(sess, source_model_file)
		target_saver.restore(sess, source_model_file)
		if not shared:
			target_saver.restore(sess, source_model_file)
	for iteration in range(nb_steps):
		indices_s = np.random.randint(0, Xs_trn.shape[0]-1, batch_size)
		batch_s = Xs_trn[indices_s,:]
		batch_ys = ys_trn[indices_s,:]
		indices_t = np.random.randint(0, Xt_trn.shape[0]-1, batch_size)
		batch_t = Xt_trn[indices_t,:]
		_, D_loss = sess.run([disc_step, -disc_loss], feed_dict={xs: batch_s, xt: batch_t, is_training: False, dis_training: True})
		_, G_loss = sess.run([gen_step, gen_loss], feed_dict={xs: batch_s, xt: batch_t, is_training: True, dis_training: False})
		_, sC_loss = sess.run([src_clf_step, src_clf_loss], feed_dict={xs: batch_s, ys: batch_ys, is_training: True, dis_training: False})
		_, M_loss = sess.run([mmd_trn_ops, mmd_loss], feed_dict={xs: batch_s, xt: batch_t, is_training: True, dis_training: False})
		if nb_trg_labels > 0:
			indices_tl = np.random.randint(0, 2*nb_trg_labels-1, 100)
			batch_xt_l, batch_yt_l = Xt_trn_l[indices_tl, :], yt_trn_l[indices_tl, :]# 			_, G_loss, sC_loss, tC_loss, trg_digit = sess.run([gen_step, gen_loss, src_clf_loss, trg_clf_loss, target_logit_l], feed_dict={xs: batch_s, xt: batch_t, ys: batch_ys, xt1:batch_xt_l, yt1:batch_yt_l, is_training: True, dis_training: False})
			_, tC_loss, trg_digit = sess.run([trg_clf_step, trg_clf_loss, target_logit_l], feed_dict={xt1:batch_xt_l, yt1:batch_yt_l, is_training: True, dis_training: False})
			train_target_stat = np.exp(trg_digit)
			train_target_AUC = roc_auc_score(batch_yt_l, train_target_stat)
		if iteration%40 == 0:
			test_source_logit = source_logit.eval(session=sess,feed_dict={xs:Xs_tst, is_training: False, dis_training: False})
			test_source_stat = np.exp(test_source_logit)
			test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
			src_test_list.append(test_source_AUC)
			train_source_logit = src_logits.eval(session=sess,feed_dict={xs:batch_s, is_training: False, dis_training: False})
			train_target_logit = trg_logits.eval(session=sess,feed_dict={xt:batch_t, is_training: False, dis_training: False})
# 			domain_preds = np.concatenate([train_source_logit, train_target_logit], axis = 0) > 0
# 			domain_labels = np.concatenate([np.ones(train_source_logit.shape), np.zeros(train_target_logit.shape)])
# 			domain_acc = np.sum(domain_preds == domain_labels)/domain_preds.shape[0]
# 			dom_acc_list.append(domain_acc)
			test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst, is_training: False, dis_training: False})
			test_target_stat = np.exp(test_target_logit)
			test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
			val_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_val, is_training: False, dis_training: False})
			val_target_stat = np.exp(val_target_logit)
			val_target_AUC = roc_auc_score(yt_val, val_target_stat)
			test_auc_list.append(test_target_AUC)
			val_auc_list.append(val_target_AUC)
			G_loss_list.append(G_loss)
			D_loss_list.append(D_loss); mmd_loss_list.append(M_loss)
			sC_loss_list.append(sC_loss)
			# save results
			np.savetxt(os.path.join(DA_model_folder,'test_auc.txt'), test_auc_list)
			np.savetxt(os.path.join(DA_model_folder,'val_auc.txt'), val_auc_list)
# 			np.savetxt(os.path.join(DA_model_folder,'dom_acc.txt'), dom_acc_list)
			np.savetxt(os.path.join(DA_model_folder,'D_loss.txt'),D_loss_list)
			np.savetxt(os.path.join(DA_model_folder,'G_loss.txt'),G_loss_list)
			np.savetxt(os.path.join(DA_model_folder,'MMD_loss.txt'),mmd_loss_list)
			np.savetxt(os.path.join(DA_model_folder,'src_clf_loss.txt'),sC_loss_list)
			# print and plot results
			print_block(symbol = '-', nb_sybl = 60)
			print_yellow(os.path.basename(DA_model_folder))
			if nb_trg_labels > 0:
				train_auc_list.append(train_target_AUC)
				tC_loss_list.append(tC_loss)
				np.savetxt(os.path.join(DA_model_folder,'train_auc.txt'), train_auc_list)
				np.savetxt(os.path.join(DA_model_folder,'trg_clf_loss.txt'),tC_loss_list)
				print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}, T-train {2:.4f}, S-test: {3:.4f}'.format(test_target_AUC, val_target_AUC, train_target_AUC, test_source_AUC))
				print_yellow('Loss: D:{0:.4f}, G:{1:.4f}, MMD:{2:.4f} S:{3:.4f}, T:{4:.4f}, Iter:{5:}'.format(D_loss, G_loss, M_loss, sC_loss, tC_loss, iteration))
				plot_LOSS(DA_model_folder+'/loss_{}.png'.format(DA_model_name), mmd_loss_list, sC_loss_list, tC_loss_list)
				plot_loss(DA_model_folder, D_loss_list, G_loss_list, DA_model_folder+'/adver_{}.png'.format(DA_model_name))
# 				plot_AUCs_DomACC(DA_model_folder+'/AUC_dom_{}.png'.format(DA_model_name), train_auc_list, val_auc_list, test_auc_list, dom_acc_list)
				plot_src_trg_AUCs(DA_model_folder+'/AUC_src_{}.png'.format(DA_model_name), train_auc_list, val_auc_list, test_auc_list, src_test_list)
				plot_AUCs(DA_model_folder+'/AUC_{}.png'.format(DA_model_name), train_auc_list, val_auc_list, test_auc_list)
			else:
				print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}, S-test: {2:.4f}'.format(test_target_AUC, val_target_AUC, test_source_AUC))
				print_yellow('Loss: D:{0:.4f}, G:{1:.4f}, MMD:{2:.4f} S:{3:.4f}, Iter:{4:}'.format(D_loss, G_loss, M_loss, sC_loss, iteration))
				plot_loss(DA_model_folder, mmd_loss_list, sC_loss_list, DA_model_folder+'/loss_{}.png'.format(DA_model_name))
				plot_loss(DA_model_folder, D_loss_list, G_loss_list, DA_model_folder+'/adver_{}.png'.format(DA_model_name))
# 				plot_auc_dom_acc_iterations(test_auc_list, val_auc_list, dom_acc_list, DA_model_folder+'/AUC_dom_{}.png'.format(DA_model_name))
				plot_auc_iterations(test_auc_list, val_auc_list, DA_model_folder+'/AUC_{}.png'.format(DA_model_name))
				plot_src_trg_auc_iterations(test_auc_list, val_auc_list, src_test_list, DA_model_folder+'/AUC_src_{}.png'.format(DA_model_name))
# 			D_grad_list1.append(np.mean(np.square(D_grads[-1]))); D_grad_list2.append(np.mean(np.square(D_grads[0])))
# 			G_grad_list1.append(np.mean(np.square(G_grads[-1]))); G_grad_list2.append(np.mean(np.square(G_grads[0])))
# 			print_yellow('grad: G_1 {0:.7f} G_2 {1:.7f} D_1 {2:.7f} D_2 {3:.7f}'.format(np.mean(np.square(D_grads[-1])), np.mean(np.square(D_grads[0])), np.mean(np.square(G_grads[-1])), np.mean(np.square(G_grads[0]))))
# 			plot_gradients(DA_model_folder + '/grad-{}.png'.format(DA_model_name), D_grad_list1, D_grad_list2, G_grad_list1, G_grad_list2, fig_size = (10,10))
			# save models
# 			if iteration%100==0:
# 				target_saver.save(sess, DA_model_folder +'/target', global_step= iteration)
			if best_val_auc < val_target_AUC:
				best_val_auc = val_target_AUC
				target_saver.save(sess, DA_model_folder+'/target_best')
				np.savetxt(os.path.join(DA_model_folder,'test_stat.txt'), test_target_stat)
				np.savetxt(os.path.join(DA_model_folder,'test_best_auc.txt'), [test_target_AUC])
				print_red('Update best:'+DA_model_folder)
			if iteration%1000 == 0:
				source_feat = h_src.eval(session=sess, feed_dict = {xs: Xs_tst, is_training: False, dis_training: False}); target_feat = h_trg.eval(session=sess, feed_dict = {xt: Xt_tst, is_training: False, dis_training: False})
				plot_feature_pair_dist(DA_model_folder+'/feat_{}.png'.format(DA_model_name), np.squeeze(source_feat), np.squeeze(target_feat), ys_tst, yt_tst, ['source', 'target'])