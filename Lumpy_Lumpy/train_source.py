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
	ax.legend(['Train','Val','Test'])
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
	ax.hist(y, **kwargs, color='r', label='SP')
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
parser.add_argument("--model_type", type = str, default = 'source')
parser.add_argument("--lr", type = float, default = 1e-5)
parser.add_argument("--bz", type = int, default = 300)
parser.add_argument("--train", type = int, default = 100000)
parser.add_argument("--nb_cnn", type = int, default = 4)
parser.add_argument("--fc_layer", type = int, default = 128)
parser.add_argument("--bn", type = str2bool, default = False)
parser.add_argument("--h", type = float, default = 40)
parser.add_argument("--blur", type = float, default = 0.5)
parser.add_argument("--noise", type = float, default = 10)
parser.add_argument("--valid", type = int, default = 100)
parser.add_argument("--test", type = int, default = 200)
parser.add_argument("--iters", type = int, default = 10000)

args = parser.parse_args()
print(args)

gpu = args.gpu
docker = args.docker
model_type = args.model_type
batch_size = args.bz
nb_steps = args.iters
lr = args.lr
fc_layer = args.fc_layer
bn = args.bn
h =args.h
blur = args.blur
noise = args.noise
train = args.train
valid = args.valid
test = args.test

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

if docker:
	output_folder ='/data/results/'
else:
	output_folder = 'data/results/'
print(output_folder)
data_folder = os.path.join(output_folder,'Lumpy/{}'.format(model_type))
# load source data
Xs_trn, Xs_val, Xs_tst, ys_trn, ys_val, ys_tst = load_Lumpy(docker = docker, train = train, valid = valid, test = test, height = h, blur= blur, noise = noise)
Xs_trn, Xs_val, Xs_tst = np.expand_dims(Xs_trn, axis = 3), np.expand_dims(Xs_val, axis = 3), np.expand_dims(Xs_tst, axis = 3)
ys_trn, ys_val, ys_tst = ys_trn.reshape(-1,1), ys_val.reshape(-1,1), ys_tst.reshape(-1,1)

model_name = 'Lumpy-{}-cnn-{}-fc-{}-lr-{}-bz-{}-bn-{}-T_V_T-{}k_{}_{}-{}_{}_{}-itrs-{}'\
				.format(model_type,nb_cnn,fc_layer, lr,batch_size,bn, int(train/1000), valid, test, h, blur,noise, nb_steps)
model_folder = os.path.join(data_folder, model_name); generate_folder(model_folder)

img_size = 64
xs = tf.placeholder("float", shape=[None, img_size, img_size, 1])
ys = tf.placeholder("float", shape=[None, 1])

conv_net_src, h_src, source_logit = conv_classifier(xs, nb_cnn = nb_cnn, fc_layers = [fc_layer,1],  bn = bn, scope_name = model_type)

source_vars_list = tf.trainable_variables(model_type)
source_key_list = [v.name[:-2].replace(model_type, 'base') for v in tf.trainable_variables(model_type)]
source_key_direct = {}
for key, var in zip(source_key_list, source_vars_list):
	source_key_direct[key] = var
source_saver = tf.train.Saver(source_key_direct, max_to_keep=nb_steps)

# source loss
src_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ys, logits = source_logit))
source_trn_ops = tf.train.AdamOptimizer(lr).minimize(src_clf_loss, var_list = source_vars_list)

train_loss_list, val_loss_list, test_loss_list = [], [], []
train_auc_list, val_auc_list, test_auc_list = [], [], []

best_val_auc = -np.inf
with tf.Session() as sess:
	print_block(symbol = '-', nb_sybl = 50)
	tf.global_variables_initializer().run(session=sess)
	for iteration in range(nb_steps):
		indices = np.random.randint(0, Xs_trn.shape[0]-1, batch_size)
		# train the source
		source_x = Xs_trn[indices,:]; source_y = ys_trn[indices,:]; sess.run(source_trn_ops, feed_dict={xs:source_x, ys: source_y})
		if iteration%100 == 0:
			train_loss = source_loss.eval(session=sess, feed_dict={xs:source_x, ys:source_y})
			val_loss = val_loss.eval(session=sess, feed_dict={xs:Xs_val, ys:ys_val})
			test_loss = val_loss.eval(session=sess, feed_dict={xs:Xs_tst, ys:ys_tst})
			train_stat = source_logit.eval(session=sess, feed_dict={xs:source_x}); train_auc = roc_auc_score(source_y, train_stat)
			val_stat = source_logit.eval(session=sess, feed_dict={xs:Xs_val}); val_auc = roc_auc_score(ys_val, val_stat)
			test_stat = source_logit.eval(session=sess, feed_dict={xs:Xt_tst}); test_auc = roc_auc_score(yt_tst, test_stat)
			train_loss_list, train_auc_list = np.append(train_loss_list, train_loss), np.append(train_auc_list, train_auc)
			val_loss_list, val_auc_list = np.append(val_loss_list, val_loss), np.append(val_auc_list, val_auc)
			test_loss_list, test_auc_list = np.append(test_loss_list, test_loss), np.append(test_auc_list, test_auc)
			np.savetxt(model_folder+'/train_loss.txt', train_loss_list);np.savetxt(model_folder+'/train_auc.txt', train_auc_list)
			np.savetxt(model_folder+'/val_loss.txt', val_loss_list);np.savetxt(model_folder+'/val_auc.txt', val_auc_list)
			np.savetxt(model_folder+'/test_loss.txt', test_loss_list);np.savetxt(model_folder+'/test_auc.txt', test_auc_list)
			plot_LOSS(model_folder+'/loss-{}.png'.format(model_name), train_loss_list, val_loss_list, test_loss_list)
			plot_AUC(model_folder+'/auc-{}.png'.format(model_name), train_auc_list, val_auc_list, test_auc_list)
			if best_val_auc < trg_val_auc:
				best_val_auc = trg_val_auc
				np.savetxt(model_folder+'/best_stat.txt', test_stat)
				target_saver.save(sess, model_folder +'/best')
				plot_hist(model_folder +'/hist-{}.png'.format(model_name), test_stat[:int(len(test_stat)/2)], test_stat[int(len(test_stat)/2):])
