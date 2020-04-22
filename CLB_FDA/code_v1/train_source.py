import tensorflow as tf

import numpy as np
import os
import argparse

from load_data import *

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.system('mkdir {}'.format(folder))

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


## input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num", type=int)
parser.add_argument("--lr", type = float)
parser.add_argument("--nb_train", type = int)
parser.add_argument("--noise", type = float)
parser.add_argument("--sig_rate", type = float)
parser.add_argument("--bz", type = int)
parser.add_argument("--optimizer", type = str)
parser.add_argument("--nb_steps", type = int, default = 100000)

args = parser.parse_args()
gpu_num = args.gpu_num
lr = args.lr
nb_train = args.nb_train
noise = args.noise
sig_rate = args.sig_rate
batch_size = args.bz
optimizer = args.optimizer
# nb_train = 100000
# sig_rate = 0.035
# noise = 2
# batch_size = 200
# lr = 5e-5
l2_param = 1e-5
bn_training = True
num_steps = args.nb_steps
# optimizer="Adam"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

## load data
X_trn, X_val, X_tst, y_trn, y_val, y_tst = load_source(train = nb_train, sig_rate = sig_rate) 							# train, valid, test data
X_val, X_tst = np.random.RandomState(0).normal(X_val, noise), np.random.RandomState(1).normal(X_tst, noise) 			# add noise
X_val, X_tst = (X_val-np.min(X_val))/(np.max(X_val)-np.min(X_val)), (X_tst-np.min(X_tst))/(np.max(X_tst)-np.min(X_tst)) # data normalization
X_val, X_tst = np.expand_dims(X_val, axis = 3), np.expand_dims(X_tst, axis = 3)
y_val, y_tst = y_val.reshape(-1,1), y_tst.reshape(-1,1)

model_root_folder = 'data/CLB' 	# dataset
generate_folder(model_root_folder)

direct = os.path.join(model_root_folder,'noise-{}-trn-{}-sig-{}-bz-{}-lr-{}-{}-{}k'.format(noise, nb_train, sig_rate, batch_size, lr, optimizer, num_steps/1000))
generate_folder(direct)
direct_st = direct+'/statistics'
generate_folder(direct_st)

## create graph
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

saver = tf.train.Saver(max_to_keep=num_steps)

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_, logits = pred_logit))
all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
total_loss = cross_entropy + l2_loss

# Optimizer
if optimizer=="Adam":
  train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
else:
  train_op= tf.train.GradientDescentOptimizer(lr).minimize(total_loss)

## network training
train_loss = []
train_auc = []

test_loss = []
test_auc = []

val_loss = []
val_auc = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i_batch in range(num_steps):
		# generate a batch
		ii = int(i_batch%(nb_train*2/batch_size))
		if ii ==0:
			shuff = np.random.permutation(nb_train*2)
		shuff_batch = shuff[ii*batch_size:(1+ii)*batch_size]
		batch_x = X_trn[shuff_batch,:]
		batch_y = y_trn[shuff_batch].reshape(-1,1)
		batch_x = np.random.normal(batch_x, noise)
		batch_x = (batch_x - np.min(batch_x))/(np.max(batch_x)-np.min(batch_x))
		batch_x = np.expand_dims(batch_x, axis = 3)
		# train the model on a batch
		sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})
		if i_batch % 100 == 0:
			print('>>>>>>The {}-th batch >>>'.format(i_batch))
			train_logit = pred_logit.eval(session=sess,feed_dict={x:batch_x})
			train_loss = np.append(train_loss, total_loss.eval(session=sess, feed_dict={x:batch_x, y_:batch_y}))
			train_stat = np.exp(train_logit)
			train_auc = np.append(train_auc,roc_auc_score(batch_y, train_stat))

			test_logit = pred_logit.eval(session=sess,feed_dict={x:X_tst})
			test_loss = np.append(test_loss,total_loss.eval(session=sess, feed_dict={x:X_tst, y_:y_tst}))
			test_stat = np.exp(test_logit)
			test_auc = np.append(test_auc,roc_auc_score(y_tst, test_stat))

			val_logit = pred_logit.eval(session=sess,feed_dict={x:X_val})
			val_loss = np.append(val_loss,total_loss.eval(session=sess,feed_dict={x:X_tst, y_:y_val}))
			val_stat = np.exp(val_logit)
			val_auc = np.append(val_auc, roc_auc_score(y_val, val_stat))

			print_green('AUC: train {0:0.4f}, val {1:.4f}, test {2:.4f}; loss: train {3:.4f}, val {4:.4f}, test {5:.4f}'.format(train_auc[-1],
				val_auc[-1], test_auc[-1], train_loss[-1], val_loss[-1], test_loss[-1]))
							
			# save the model and results
			model_folder = os.path.join(model_root_folder,os.path.basename(direct))
			generate_folder(model_folder)
			generate_folder(model_folder)
			saver.save(sess, model_folder+'/model', global_step=i_batch)
			print(model_folder)
			# save results
			np.savetxt(direct+'/training_auc.txt',train_auc)
			np.savetxt(direct+'/testing_auc.txt',test_auc)
			np.savetxt(direct+'/training_loss.txt',train_loss)
			np.savetxt(direct+'/testing_loss.txt',test_loss)

			np.savetxt(direct+'/val_loss.txt',val_loss)
			np.savetxt(direct+'/val_auc.txt',val_auc)
			np.savetxt(direct_st+'/statistics_'+str(i_batch)+'.txt',test_stat)
