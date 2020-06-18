import os
import numpy as np
import glob

result_root_folder = 'experiments/total_mmd_tf'
result_folders = glob.glob(result_root_folder+'/*')

for result_folder in result_folders:
	if os.path.isdir(result_folder):
		if os.path.exists(result_folder+'/test_stat.txt'):
			stat_arr = np.loadtxt(result_folder+'/test_stat.txt')
		elif os.path.exists(result_folder+'/best_test_stat_100.txt'):
			stat_arr = np.loadtxt(result_folder+'/best_test_stat_100.txt')
		else:
			continue
		if len(stat_arr) == 0:
			continue
		stat_arr = stat_arr.flatten()
		stat_len = len(stat_arr)
		t0 = stat_arr[:int(stat_len/2)]
		t1 = stat_arr[int(stat_len/2):]
		if not len(t0) == len(t1):
			continue
		fit_input_file = result_folder + '/roc_fit_input.txt'
		if os.path.exists(fit_input_file):
			os.system('rm -f {}'.format(fit_input_file))
		with open(fit_input_file, 'w+') as f:
			f.write('LABROC\n')
			f.write('Large\n')
			for i in range(len(t0)):
				f.write('{0:.6f}\n'.format(t0[i]))
			f.write('*\n')
			for i in range(len(t1)):
				f.write('{0:.6f}\n'.format(t1[i]))
			f.write('*\n')