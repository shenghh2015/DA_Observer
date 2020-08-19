import os
import numpy as np
from load_data import *
from skimage import io

def generate_folder(folder):
	import os
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))

docker = True

if docker:
	output_folder ='/data/datasets/image_trans/'
else:
	output_folder = 'data'

noise = 2.0
sig_rate = 0.035
dataset = 'total'
nb_source = 100000

Xs_trn, Xs_val, Xs_tst, ys_trn, ys_val, ys_tst = load_source(train = nb_source, sig_rate = sig_rate)
Xs_trn, Xs_val, Xs_tst = np.random.RandomState(2).normal(Xs_trn, noise), np.random.RandomState(0).normal(Xs_val, noise), np.random.RandomState(1).normal(Xs_tst, noise)
Xs_trn, Xs_val, Xs_tst = (Xs_trn-np.min(Xs_trn))/(np.max(Xs_trn)-np.min(Xs_trn)), (Xs_val-np.min(Xs_val))/(np.max(Xs_val)-np.min(Xs_val)), (Xs_tst-np.min(Xs_tst))/(np.max(Xs_tst)-np.min(Xs_tst))

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

## save images
data_pools = ['FDA', 'CLB']
subsets = ['train', 'val', 'test']
# data_dir = data_pools[0]
# subset = subsets[0]

for data_dir in data_pools:
	for subset in subsets:
		if data_dir == 'FDA':
			if subset == 'train':
				X, y = Xt_trn, yt_trn
			elif subset == 'val':
				X, y = Xt_val, yt_val
			elif subset == 'test':
				X, y = Xt_tst, yt_tst
		else:
			if subset == 'train':
				X, y = Xs_trn, ys_trn
			elif subset == 'val':
				X, y = Xs_val, ys_val
			elif subset == 'test':
				X, y = Xs_tst, ys_tst
		print('dataset: {}, subset {}'.format(data_dir, subset))
		x_dir = os.path.join(output_folder, data_dir, subset); generate_folder(x_dir)
		y_file = os.path.join(output_folder, data_dir, '{}_labels.txt'.format(subset))
		np.savetxt(y_file, y)
		for i in range(X.shape[0]):
			x = X[i,:,:]; x = np.uint8(255*x)
			io.imsave(x_dir+'/{:06d}.png'.format(i), x)
			if i%10000 == 0:
				print('{} saved ...'.format(i))

		images = []
		for i in range(2):
			x_ = io.imread(x_dir+'/{:06d}.png'.format(i))
			images.append(x_)
		images = np.stack(images); print('scale: min {:.4f} max {:.4f}'.format(images.min(), images.max()))