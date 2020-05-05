import os
import numpy
import glob

result_root_folder = 'experiment/total_mmd'
result_folders = glob.glob(result_root_folder+'/*')

for result_folder in result_folders if os.path.isdir(result_folder) :
	if os.path.exists(result_folder+'/test_stat.txt'):
		stat_arr = np.loadtxt(result_folder+'/test_stat.txt')
	elif os.path.exists(result_folder+'/best_test_stat_100.txt'):
		stat_arr = np.loadtxt(result_folder+'/best_test_stat_100.txt')
	else:
		continue
	fit_input_file = result_folder + '/roc_fit_input.txt'
	if os.path.exists(fit_input_file):
		os.system('rm -f {}'.format(fit_input_file))
	with open(fit_input_file, 'w+') as f:
		f.write('LABROC\n')
		f.write('Large\n')
		