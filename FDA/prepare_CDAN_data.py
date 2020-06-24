import os
import glob
import numpy as np

def load_source(train = 80000, valid = 400, test = 400, sig_rate = 0.035):
# 	train = 80000
# 	valid = 400
# 	test = 400
# 	sig_rate = 0.035
    sig_file = '/shared/planck/Phantom/Breast_Xray/FDA_signals/hetero_sig.dat'
    CLB_file = '/shared/rsaas/shenghua/CLB/CLB_128N_400000IM.npy'
# 	sig_file = os.path.join(dataset_folder, 'FDA_signals/hetero_sig.dat')
# 	CLB_file = os.path.join(dataset_folder, 'CLB/CLB_128N_400000IM.npy')
    sig = np.fromfile(sig_file, dtype = np.float32).reshape(109,109)
    data = np.load(CLB_file, mmap_mode='r')
    X = data[:,:,0:train+valid+test]
    X = np.transpose(np.reshape(X, [128, 128, train+valid+test], order='F'))
    X_SA_trn, X_SA_val, X_SA_tst = X[:train, 64-54:64+55,64-54:64+55], X[train:train+valid, 64-54:64+55,64-54:64+55],\
        X[train+valid:train+valid+test, 64-54:64+55,64-54:64+55]
    X_SP_trn, X_SP_val, X_SP_tst = X_SA_trn + sig * sig_rate, X_SA_val + sig * sig_rate, X_SA_tst + sig * sig_rate
    X_trn, X_val, X_tst = np.concatenate([X_SA_trn, X_SP_trn]), np.concatenate([X_SA_val, X_SP_val]), np.concatenate([X_SA_tst, X_SP_tst])
    y_trn = np.concatenate([np.zeros((train,1)), np.ones((train,1))]).flatten()
    y_val = np.concatenate([np.zeros((valid,1)), np.ones((valid,1))]).flatten()
    y_tst = np.concatenate([np.zeros((test,1)), np.ones((test,1))]).flatten()

    return X_trn, X_val, X_tst, y_trn, y_val, y_tst

def load_target(dataset = 'total', train = 80000, valid = 400, test = 400):
# 	dataset = 'total'
# 	train = 80000
# 	valid = 400
# 	test = 400
# 	X_SA = np.load('/shared/planck/Phantom/Breast_Xray/FDA_DM_ROIs/npy_dataset/{}_SA.npy'.format(dataset))
# 	X_SP = np.load('/shared/planck/Phantom/Breast_Xray/FDA_DM_ROIs/npy_dataset/{}_SP.npy'.format(dataset))
    dataset_folder = '/shared/planck/Phantom/Breast_Xray/'
    if dataset == 'dense':
        offset_valid = 7100
    elif dataset == 'hetero':
        offset_valid = 36000
    elif dataset == 'scattered':
        offset_valid = 33000
    elif dataset == 'fatty':
        offset_valid = 9000
    elif dataset == 'total':
        offset_valid = 85000
    offset_test = 400 + offset_valid
    X_SA = np.load(os.path.join(dataset_folder, 'FDA_DM_ROIs/npy_dataset/{}_SA.npy'.format(dataset)))
    X_SP = np.load(os.path.join(dataset_folder, 'FDA_DM_ROIs/npy_dataset/{}_SP.npy'.format(dataset)))
    X_SA_trn, X_SA_val, X_SA_tst = X_SA[:train,:], X_SA[offset_valid:offset_valid+valid,:], X_SA[offset_test:offset_test+test,:]
    X_SP_trn, X_SP_val, X_SP_tst = X_SP[:train,:], X_SP[offset_valid:offset_valid+valid,:], X_SP[offset_test:offset_test+test,:]
    X_trn, X_val, X_tst = np.concatenate([X_SA_trn, X_SP_trn]), np.concatenate([X_SA_val, X_SP_val]), np.concatenate([X_SA_tst, X_SP_tst])
    y_trn = np.concatenate([np.zeros((train,1)), np.ones((train,1))]).flatten()
    y_val = np.concatenate([np.zeros((valid,1)), np.ones((valid,1))]).flatten()
    y_tst = np.concatenate([np.zeros((test,1)), np.ones((test,1))]).flatten()
    print('---- Dataset Summary: {} ----'.format(dataset))
    print(' -all SA {}, SP {}'.format(X_SA.shape[0], X_SP.shape[0]))
    print(' -trn SA {}, SP {}'.format(X_SA_trn.shape[0], X_SP_trn.shape[0]))
    print(' -val SA {}, SP {}'.format(X_SA_val.shape[0], X_SP_val.shape[0]))
    print(' -val SA {}, SP {}'.format(X_SA_tst.shape[0], X_SP_tst.shape[0]))
    # 	print('\n')
    return X_trn, X_val, X_tst, y_trn, y_val, y_tst

# load source data
noise = 2.0
sig_rate = 0.035
nb_source = 10000
Xs_trn, Xs_val, Xs_tst, ys_trn, ys_val, ys_tst = load_source(train = nb_source, sig_rate = sig_rate)
Xs_trn, Xs_val, Xs_tst = np.random.RandomState(2).normal(Xs_trn, noise), np.random.RandomState(0).normal(Xs_val, noise), np.random.RandomState(1).normal(Xs_tst, noise)
# load target data
dataset = 'total'
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

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

from skimage import io

dataset_folder = '/home/sh38/domain_adaptation/CDAN/data/fda/'
source_folder = dataset_folder+'clb/'; target_folder = dataset_folder +'fda/'
generate_folder(source_folder); generate_folder(target_folder)
indices_s = np.random.permutation(Xs.shape[0])
print(indices_s.shape)
data_list_file = dataset_folder+'clb_list.txt'
if os.path.exists(data_list_file):
    os.system('rm -f {}'.format(data_list_file))
print(Xs.shape[0])

with open(data_list_file, 'w+') as f:
    for i in range(Xs.shape[0]):
        image = Xs[indices_s[i],:,:]; rgb_img = np.stack([image,image,image],axis=2);
        rgb_img = np.uint8(rgb_img-rgb_img.min()/(rgb_img.max()-rgb_img.min())*255.)
        image_file_path = source_folder+'{:05d}.png'.format(indices_s[i])
        io.imsave(image_file_path,rgb_img)
        f.write(image_file_path+ ' {}\n'.format(int(ys[indices_s[i]])))

indices_t = np.random.permutation(Xt.shape[0])
print(indices_t.shape)

data_list_file = dataset_folder+'fda_list.txt'
if os.path.exists(data_list_file):
    os.system('rm -f {}'.format(data_list_file))
print(Xt.shape[0])

with open(data_list_file, 'w+') as f:
    for i in range(Xt.shape[0]):
        image = Xt[indices_t[i],:,:]; rgb_img = np.stack([image,image,image],axis=2);
        rgb_img = np.uint8(rgb_img-rgb_img.min()/(rgb_img.max()-rgb_img.min())*255.)
        image_file_path = target_folder+'{:05d}.png'.format(indices_t[i])
        io.imsave(image_file_path,rgb_img)
        f.write(image_file_path+ ' {}\n'.format(int(ys[indices_t[i]])))