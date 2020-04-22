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


import os
from load_data import *
from models import *

def save_model_epoch_idx(model,model_name,epoch_idx):
	generate_folder(model_name)
	# serialize model to YAML
	model_yaml = model.to_yaml()
	model_path = model_name+"/model.yaml"
	if not os.path.exists(model_path):
		with open(model_path, "w") as yaml_file:
			yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights(model_name+"/weights_{}.h5".format(epoch_idx))

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.system('mkdir {}'.format(folder))

def train(gpu_num = 0, nb_train = 10000, dataset = 'total', lr = 1e-4):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
	batch_size = 200
	learning_rate = lr
	optimizer = 'Adam'
	initializer='truncated_normal'
	nb_train = nb_train

	# train, valid, test data
	X_trn, X_val, X_tst, y_trn, y_val, y_tst = load_target(dataset = 'total', train = nb_train)
	
	# add noise
# 	noise = noise
# 	X_val, X_tst = np.random.RandomState(0).normal(X_val, noise), np.random.RandomState(1).normal(X_tst, noise)
	# data normalization
	X_val, X_tst = (X_val-np.min(X_val))/(np.max(X_val)-np.min(X_val)), (X_tst-np.min(X_tst))/(np.max(X_tst)-np.min(X_tst))
	X_val, X_tst = np.expand_dims(X_val, axis = 3), np.expand_dims(X_tst, axis = 3)
	y_val, y_tst = y_val.reshape(-1,1), y_tst.reshape(-1,1)
	# model folder
	model_root_folder = 'BCE_FDA'
	generate_folder(model_root_folder)

	direct = os.path.join(model_root_folder,'dataset-{}-train-{}-lr-{}'.format(dataset, nb_train, learning_rate))
	generate_folder(direct)
	direct_st = direct+'/statistics'
	generate_folder(direct_st)

	# create a model
	bn = True
	model = buildCNNClassifierModel(input_shape = (109,109,1), kernel_initializer = initializer, bn = bn)
	optimizer = Adam(lr = lr)
	loss_function = 'binary_crossentropy'
	model.compile(optimizer = optimizer, loss = loss_function)

	train_loss = []
	train_auc = []

	test_loss = []
	test_auc = []

	val_loss = []
	val_auc = []
	for iter in range(100000):
		ii = int(iter%(nb_train*2/batch_size))
		if ii ==0:
		    shuff = np.random.permutation(nb_train*2)
		shuff_batch = shuff[ii*batch_size:(1+ii)*batch_size]
		batch_x = X_trn[shuff_batch,:]
		batch_y = y_trn[shuff_batch].reshape(-1,1)
# 		tmp0 = h0_train[shuff_batch,:,:]
# 		tmp1 = tmp0 + sig
		# train_data = np.concatenate([h0_train,h1_train], axis = 0)
		# train_data = np.random.normal(train_data, 20)
		# train_data = np.concatenate([h0_train,h1_train], axis = 0)
		# train_data = np.random.normal(train_data, 20)
		# train_data = np.expand_dims(train_data, axis = 3)
		# hist = model.fit(train_data, y_data, batch_size=batch_size, nb_epoch=1, verbose=1, shuffle=True)
		# trn_scores = model.predict(train_data, batch_size = batch_size)
# 		batch_x = np.concatenate([tmp0,tmp1], axis = 0)
# 		batch_x = np.random.normal(batch_x, noise)
		batch_x = (batch_x - np.min(batch_x))/(np.max(batch_x)-np.min(batch_x))
		batch_x = np.expand_dims(batch_x, axis = 3)
# 		batch_y = np.concatenate([np.zeros((batch_size,1)), np.ones((batch_size,1))])
		hist = model.train_on_batch(batch_x, batch_y)
		if iter%100 ==0:
			print('>>>>>>The {}-th batch >>>'.format(iter))
			# hist = model.fit(batch_x, y_data, batch_size=batch_size, nb_epoch=1, verbose=1, shuffle=True)
			# trn_scores = model.predict(train_data, batch_size = batch_size)
			trn_scores = model.predict(batch_x)
			# trn_stat = np.exp(trn_scores)/(np.exp(trn_scores)+1)
			trn_stat = trn_scores
			# train_loss.append(hist.history['loss'])
			train_loss.append(hist)
			# trn_auc = roc_auc_score(y_data, trn_stat)
			trn_auc = roc_auc_score(batch_y, trn_stat)
			train_auc.append(trn_auc)
			print('Train loss:{}, train AUC:{}'.format(train_loss[-1], train_auc[-1]))
			
			tst_los = model.evaluate(X_tst, y_tst, batch_size=batch_size)
			tst_scores = model.predict(X_tst)
			test_loss.append(tst_los)
			test_stat = tst_scores
			# test_stat = np.exp(tst_scores)/(np.exp(tst_scores)+1)
			test_auc.append(roc_auc_score(y_tst, test_stat))
			print('Test loss:{}, test AUC:{}'.format(test_loss[-1], test_auc[-1]))

			val_los = model.evaluate(X_val, y_val, batch_size=batch_size)
			val_scores = model.predict(X_val)
			val_stat = val_scores
			# val_stat = np.exp(val_scores)/(np.exp(val_scores)+1)
			val_loss.append(val_los)
			val_auc.append(roc_auc_score(y_val, val_stat))
			print('Val loss:{}, val AUC:{}'.format(val_loss[-1], val_auc[-1]))

			model_folder = os.path.join(model_root_folder,os.path.basename(direct))
			print(model_folder)
			generate_folder(model_folder)
			save_model_epoch_idx(model, model_folder,iter)
			np.savetxt(direct+'/training_auc.txt',train_auc)
			np.savetxt(direct+'/testing_auc.txt',test_auc)
			np.savetxt(direct+'/training_loss.txt',train_loss)
			np.savetxt(direct+'/testing_loss.txt',test_loss)

			np.savetxt(direct+'/val_loss.txt',val_loss)
			np.savetxt(direct+'/val_auc.txt',val_auc)
			np.savetxt(direct_st+'/statistics_'+str(iter)+'.txt',test_stat)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--nb_train", type=int)
parser.add_argument("--lr", type=float)

args = parser.parse_args()
gpu_num = args.gpu_num
dataset = args.dataset
nb_train = args.nb_train
lr = args.lr

train(gpu_num, nb_train, dataset, lr)

# if __name__ == '__main__':
# 	train()
