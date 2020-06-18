import os
import glob
import numpy as np

result_folder = '/data/analysis/'

model_folders = glob.glob(result_folder + '/*')

for folder in model_folders:
	test_auc = np.loadtxt(folder+'/test_auc.txt')
	val_auc = np.loadtxt(folder+'val_auc.txt')
	idx = np.argmax(val_auc)
	max_val = np.max(val_auc)
	select_test = test_auc[idx]
	print(os.path.basename(folder))
	print('AUC: test:{0:.2f}, val:{1:.3f}'.format(select_test, max_val))
