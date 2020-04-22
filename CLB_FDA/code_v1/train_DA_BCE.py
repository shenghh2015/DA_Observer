import numpy as np
import os
import glob
import keras

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
from models_v1 import *

def save_model_epoch_idx(model,model_name,epoch_idx):
	generate_folder(model_name)
	# serialize model to YAML
	model_yaml = model.to_yaml()
	model_path = model_name+"/target_model.yaml"
	if not os.path.exists(model_path):
		with open(model_path, "w") as yaml_file:
			yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights(model_name+"/target_model_{}.h5".format(epoch_idx))

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

def save_model(model,model_name):
	generate_folder(model_name)
	# serialize model to YAML
	model_yaml = model.to_yaml()
	model_path = model_name+"/source_model.yaml"
	if not os.path.exists(model_path):
		with open(model_path, "w") as yaml_file:
			yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights(model_name+"/source_model.h5")
	print(model_name+"/source_model.h5")

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

def set_on(model, trainalbe):
	for layer in model.layers:
		layer.trainable = trainalbe
	model.tranable = trainalbe

def print_yellow(str):
	from termcolor import colored 
	print(colored(str, 'yellow'))

def print_red(str):
	from termcolor import colored 
	print(colored(str, 'red'))

def print_block(symbol = '*', nb_sybl = 70):
	print_red(symbol*nb_sybl)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num", type=int)
parser.add_argument("--lr", type = float)
parser.add_argument("--bz", type = int)
parser.add_argument("--dis_v", type = int)
parser.add_argument("--nb_dis", type = int)

args = parser.parse_args()
gpu_num = args.gpu_num
lr = args.lr
batch_size = args.bz
dis_v = args.dis_v
nb_dis = args.nb_dis

if False:
	gpu_num = 6
	lr = 1e-4
	batch_size = 400
	dis_v = 1
	nb_dis = 5

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

#DA-lr-0.0002-clip-0.05-ndis-1-enc-1-int-orthogonal-bn-True-d-3-batchs-200-disNamebuildDiscModel_v7-decay-False-decInterval-250-trg_size-5000
# parameters related to domain adaptation model training
bn = True  ## batch normization
initializer = 'truncated_normal'
# initializer = 'orthogonal'
loss_fn = 'binary_crossentropy'
optimizer1 = Adam(lr = lr*5)
optimizer2 = Adam(lr = lr)

# hyper-parameters
noise = 2.0
sig_rate = 0.035
source_model_name = 'noise-2.0-trn-100000-sig-0.035-bz-200-lr-5e-05'
# load source data
source = 'BCE_CLB'
target = 'BCE_FDA'
source_model_file = os.path.join(source, source_model_name, 'source_weights.h5')

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

source_func_name, disc_name = 'buildCNNClassifierModel', 'buildDiscModel_v{}'.format(dis_v)

DA = 'BCE_{}-{}'.format(source, target)
generate_folder(DA)
base_model_folder = os.path.join(DA, source_model_name)
generate_folder(base_model_folder)
# copy the source weight file to the DA_model_folder
DA_model_name = 'bz-{}-lr-{}-dis_v-{}-nb_dis-{}'.format(batch_size, lr, dis_v, nb_dis)
DA_model_folder = os.path.join(base_model_folder, DA_model_name)
generate_folder(DA_model_folder)
os.system('cp -f {} {}'.format(source_model_file, DA_model_folder))

# load the source encoder part from the source model
source_model = globals()[source_func_name](input_shape = (109,109,1), kernel_initializer = initializer, bn = bn)
source_model.load_weights(DA_model_folder+'/source_weights.h5', by_name = True)
save_model(source_model, DA_model_folder)

# load the target encoder
target_model = globals()[source_func_name](input_shape = (109,109,1), kernel_initializer = initializer, bn = bn)
source_model_weight_file = os.path.join(DA_model_folder, 'source_model.h5')
target_model.load_weights(source_model_weight_file, by_name = False)

# check whether the source model is well loaded
# feat_map1 = source_model.predict(Xs_tst)
source_scores = source_model.predict(Xs_tst)
source_auc = roc_auc_score(ys_tst, source_scores)

# feat_map2 = target_model.predict(Xt_tst)
target_scores = source_model.predict(Xt_tst)
target_auc = roc_auc_score(yt_tst, target_scores)
print('>>>>>> Check the Initial Source Model Loading <<<<<<')
print('Source to Source:{0:.4f} '.format(source_auc))
print('Source to Target:{0:.4f}'.format(target_auc))

####  training
# loss = globals()[loss_fn]
disc_function = globals()[disc_name]
bn = True
discriminator = disc_function(target_model.output_shape[1:])
discriminator.trainable = True
target_model.trainable = False
discriminator.compile(optimizer = optimizer1, loss = loss_fn, metrices = ['accuracy'])
target_model_discriminator = Model(target_model.input, discriminator(target_model.output))
target_model_discriminator.compile(optimizer = optimizer2, loss = loss_fn, metrices = ['accuracy'])

D_loss_list = []
G_loss_list = []
tst_auc_list = []
val_auc_list = []
target_model_list = []
discr_list = []

target_model_folder = os.path.join(DA_model_folder,'target_model')
generate_folder(target_model_folder)
discr_folder = os.path.join(DA_model_folder,'discr')
generate_folder(discr_folder)
## save the first target encoder and discriminator
save_model_epoch_idx(target_model,target_model_folder,0)
save_discr_epoch_idx(discriminator,discr_folder,0)
target_model_list.append(os.path.join(target_model_folder,'target_model_0.h5'))
discr_list.append(os.path.join(discr_folder,'discr_0.h5'))

source_labels = np.ones((batch_size, 1))
target_labels = np.zeros((batch_size, 1))
for iter in range(100000):
	if iter > 500:
		nb_dis = 1
# 	print_block(symbol = '-', nb_sybl = 70)
# 	print(os.path.basename(DA_model_folder))
	optimizer1 = Adam(lr = lr*5)
	optimizer2 = Adam(lr = lr)
	## load the weights for source encoder, target encoder, source classifier, discriminator
	source_model = globals()[source_func_name](input_shape = (109, 109,1), kernel_initializer = initializer, bn = bn)
	target_model = globals()[source_func_name](input_shape = (109, 109,1), kernel_initializer = initializer, bn = bn)
	discriminator = disc_function(target_model.output_shape[1:])
	target_model_discriminator = Model(target_model.input, discriminator(target_model.output))
	source_model.load_weights(source_model_weight_file, by_name = False)
# 	print('Load target model file:{}'.format(os.path.basename(target_model_list[-1])))
# 	print('Load discriminator file:{}'.format(os.path.basename(discr_list[-1])))
	target_model.load_weights(target_model_list[-1], by_name = False)
	discriminator.load_weights(discr_list[-1], by_name = False)

	indices_s = np.random.randint(0, Xs_trn.shape[0], batch_size)
	batch_s = Xs_trn[indices_s,:]
	indices_t = np.random.randint(0, Xt_trn.shape[0], batch_size)
	batch_t = Xt_trn[indices_t,:]
	sfeats = source_model.predict(batch_s)
	tfeats = target_model.predict(batch_t)
	train_feats = np.concatenate([sfeats, tfeats], axis = 0)
	labels = np.concatenate([source_labels, target_labels])
	# train the discriminator
	if not iter%(nb_dis+1) == nb_dis:
# 		set_on(discriminator, True)
		discriminator.trainable = True
		discriminator.compile(optimizer = optimizer1, loss = loss_fn)
		loss_D = discriminator.train_on_batch(train_feats, labels)
		save_discr_epoch_idx(discriminator, discr_folder,iter+1)    # save the discriminator
		discr_list.append(os.path.join(discr_folder, 'discr_{}.h5'.format(iter+1)))
		if iter%(nb_dis+1) == nb_dis-1:
			D_loss_list.append(loss_D)
	else:
		# train the target encoder		
		discriminator.trainable = False
# 		set_on(discriminator, False)
# 		discriminator.compile(optimizer = optimizer1, loss = loss_fn)
		target_model_discriminator.compile(optimizer = optimizer2, loss = loss_fn)			
		loss_G = target_model_discriminator.train_on_batch(batch_t, source_labels)
		save_model_epoch_idx(target_model,target_model_folder,iter+1)
		target_model_list.append(os.path.join(target_model_folder,'target_model_{}.h5'.format(iter+1)))
		G_loss_list.append(loss_G)
		print_block(symbol = '-', nb_sybl = 70)
		print_yellow(os.path.basename(DA_model_folder))
		print_yellow('Loss: D {0:.4f}, G loss:{1:.4f}; Iter:{2}'.format(loss_D, loss_G, iter+1))
		plot_loss(DA_model_folder, D_loss_list, G_loss_list, DA_model_folder+'/loss_{}.png'.format(DA_model_name))
		tst_scores = target_model.predict(Xt_tst)
		val_scores = target_model.predict(Xt_val)
		tst_auc = roc_auc_score(yt_tst, tst_scores)
		val_auc = roc_auc_score(yt_val, val_scores)
		tst_auc_list.append(tst_auc)
		val_auc_list.append(val_auc)
		np.savetxt(os.path.join(DA_model_folder,'test_auc.txt'), tst_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'val_auc.txt'), val_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'D_loss.txt'),D_loss_list)
		np.savetxt(os.path.join(DA_model_folder,'G_loss.txt'),G_loss_list)
		plot_auc_iterations(tst_auc_list, val_auc_list, DA_model_folder+'/AUC_{}.png'.format(DA_model_name))
		print_yellow('AUC: Target-DA-Source {}'.format(tst_auc))
		print_block(symbol = '-', nb_sybl = 70)
	K.clear_session()
	gc.collect()