import os
import glob
import numpy as np

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

# load target data
dataset = 'total'; nb_target = 1000
Xt_trn, Xt_val, Xt_tst, yt_trn, yt_val, yt_tst = load_target(dataset = dataset, train = nb_target, valid = 100)

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

from skimage import io
Xt = Xt_tst; yt  = yt_tst
dataset_folder = '/home/sh38/domain_adaptation/CDAN/data/fda-2/'
target_folder = dataset_folder +'fda_tst/'
generate_folder(target_folder)

indices_t = np.random.permutation(Xt.shape[0])
print(indices_t.shape)

data_list_file = dataset_folder+'fda_tst_list.txt'
if os.path.exists(data_list_file):
    os.system('rm -f {}'.format(data_list_file))
print(Xt.shape[0])

with open(data_list_file, 'w+') as f:
    for i in range(Xt.shape[0]):
        image = Xt[indices_t[i],:,:]; rgb_img = np.stack([image,image,image],axis=2);
        rgb_img = np.uint8((rgb_img-rgb_img.min())/(rgb_img.max()-rgb_img.min())*255.)
        image_file_path = target_folder+'{:06d}.png'.format(indices_t[i])
        io.imsave(image_file_path,rgb_img)
        f.write(image_file_path+ ' {}\n'.format(int(yt[indices_t[i]])))