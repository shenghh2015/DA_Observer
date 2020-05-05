import os
import numpy as np
import glob
from termcolor import colored 
import argparse

def print_yellow(str):
	from termcolor import colored 
	print(colored(str, 'yellow'))

def print_red(str):
	from termcolor import colored 
	print(colored(str, 'red'))

def print_green(str):
	from termcolor import colored 
	print(colored(str, 'green'))

def str2bool(value):
    return value.lower() == 'true'

def print_block(symbol = '*', nb_sybl = 70):
	print_red(symbol*nb_sybl)

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, default = 'dense')
parser.add_argument("docker", type=str2bool, default = 'true')
args = parser.parse_args()
print(args)

dataset = args.dataset
docker = args.docker

if docker:
	result_folder = '/data/backup'
else:
	result_folder = '/shared2/Data_FDA_Breast/Observer/backup'

method = 'adda'
# dataset = 'dense'

## source model training
print_green('The source model profermance')
source_folder = os.path.join(result_folder, 'CLB')
source_model_folders = glob.glob(source_folder +'/*')
for source_model in source_model_folders:
	if os.path.isdir(source_model) and os.path.exists(source_model+'/source-best.meta'):
		print_red(os.path.basename(source_model))
		#load the AUC performance
		val_auc = np.loadtxt(source_model+'/val_auc.txt')
		test_auc = np.loadtxt(source_model+'/testing_auc.txt')
		select_Idx = np.argmax(val_auc)
		print_yellow('AUC: Best Test {0:.4f},  Val {1:.4f}'.format(test_auc[select_Idx], val_auc[select_Idx]))

## directly train the model

## Transfer learning (TF)
def classify_categories(base_model_folder, method = 'TF', dataset = 'dense'):
# 	method = 'TF'
# 	print('Base Model: {}'.format(os.path.basename(base_model_folder)))
# 	target_models = [v for v in glob.glob(base_model_folder+'/*') if os.path.isdir(v) and method in os.path.basename(v) and dataset in os.path.basename(v)]
	if dataset == 'dense':
		target_models = [v for v in glob.glob(base_model_folder+'/*') if os.path.isdir(v) and method in os.path.basename(v) and 'dense' in os.path.basename(v)]
	elif dataset == 'total':
		target_models = [v for v in glob.glob(base_model_folder+'/*') if os.path.isdir(v) and method in os.path.basename(v) and (not 'dense' in os.path.basename(v))]
	t0_list, t70_list, t100_list, t200_list, t300_list, t400_list, t500_list = [],[],[],[],[], [], [], []
	for i in range(len(target_models)):
		target_model = target_models[i]
		model_name = os.path.basename(target_model)
# 		print_red(model_name)
		if 'labels-0' in model_name or (not 'labels' in model_name):
			t0_list.append(target_model)
		if 'labels-70' in model_name:
			t70_list.append(target_model)		
		if 'labels-100' in model_name:
			t100_list.append(target_model)
		if 'labels-200' in model_name:
			t200_list.append(target_model)
		if 'labels-300' in model_name:
			t300_list.append(target_model)
		if 'labels-400' in model_name:
			t400_list.append(target_model)
		if 'labels-500' in model_name:
			t500_list.append(target_model)
		if 'labels-1000' in model_name:
			t500_list.append(target_model)

	return t0_list, t70_list, t100_list, t200_list, t300_list, t400_list, t500_list, t1000_list

def present_auc(model_list):
	for m in model_list:
# 		if os.path.exists(m+'/target_best.meta') and os.path.exists(m+'/val_auc.txt'):
		if os.path.exists(m+'/val_auc.txt'):
			model_name = os.path.basename(m)
			val_auc = np.loadtxt(m+'/val_auc.txt')
			test_auc = np.loadtxt(m+'/test_auc.txt')
			if len(val_auc.shape)>0:
				if len(val_auc) == len(test_auc) and len(val_auc) > 0:
					select_Idx = np.argmax(val_auc)
					print_red(model_name)
					print_yellow('AUC: Best Test {0:.4f},  Val {1:.4f}'.format(test_auc[select_Idx], val_auc[select_Idx]))

print_green('Transfer Learning')
DA_folder = os.path.join(result_folder, 'CLB-FDA')
base_model_folders = glob.glob(DA_folder +'/*')
for base_model_folder in base_model_folders:
	if os.path.isdir(base_model_folder):
		_, t70_list, t100_list, t200_list, t300_list, t400_list, t500_list, 1000_list = classify_categories(base_model_folder, 'TF', dataset)
		print('Target label amount: {}'.format(70))
		present_auc(t70_list)
		print('Target label amount: {}'.format(100))
		present_auc(t100_list)
		print('Target label amount: {}'.format(200))
		present_auc(t200_list)
		print('Target label amount: {}'.format(300))
		present_auc(t300_list)
		print('Target label amount: {}'.format(400))
		present_auc(t400_list)
		print('Target label amount: {}'.format(500))
		present_auc(t500_list)
		print('Target label amount: {}'.format(1000))
		present_auc(t1000_list)

## ADDA
print_green('Adversarial Domain Adaptation')
DA_folder = os.path.join(result_folder, 'CLB-FDA')
base_model_folders = glob.glob(DA_folder +'/*')
for base_model_folder in base_model_folders:
	if os.path.isdir(base_model_folder):
		t0_list, t70_list, t100_list, t200_list, t300_list, t400_list, t500_list, t1000_list = classify_categories(base_model_folder, 'ADDA', dataset)
		print('Target label amount: {}'.format(0))
		present_auc(t0_list)
		print('Target label amount: {}'.format(70))
		present_auc(t70_list)
		print('Target label amount: {}'.format(100))
		present_auc(t100_list)
		print('Target label amount: {}'.format(200))
		present_auc(t200_list)
		print('Target label amount: {}'.format(300))
		present_auc(t300_list)
		print('Target label amount: {}'.format(400))
		present_auc(t400_list)
		print('Target label amount: {}'.format(500))
		present_auc(t500_list)
		print('Target label amount: {}'.format(1000))
		present_auc(t1000_list)

## mmd
print_green('Maxim Mean Dsicrepancy')
DA_folder = os.path.join(result_folder, 'CLB-FDA')
base_model_folders = glob.glob(DA_folder +'/*')
for base_model_folder in base_model_folders:
	if os.path.isdir(base_model_folder):
		t0_list, t70_list, t100_list, t200_list, t300_list, t400_list, t500_list, t1000_list = classify_categories(base_model_folder, 'mmd', dataset)
		print('Target label amount: {}'.format(0))
		present_auc(t0_list)
		print('Target label amount: {}'.format(70))
		present_auc(t70_list)
		print('Target label amount: {}'.format(100))
		present_auc(t100_list)
		print('Target label amount: {}'.format(200))
		present_auc(t200_list)
		print('Target label amount: {}'.format(300))
		present_auc(t300_list)
		print('Target label amount: {}'.format(400))
		present_auc(t400_list)
		print('Target label amount: {}'.format(500))
		present_auc(t500_list)
		print('Target label amount: {}'.format(1000))
		present_auc(t1000_list)


## naively-trained target model
def present_auc(model_list):
	for m in model_list:
# 		if os.path.exists(m+'/target_best.meta') and os.path.exists(m+'/val_auc.txt'):
		if os.path.exists(m+'/val_auc.txt'):
			model_name = os.path.basename(m)
			val_auc = np.loadtxt(m+'/val_auc.txt')
			test_auc = np.loadtxt(m+'/testing_auc.txt')
			if len(val_auc.shape)>0:
				if len(val_auc) == len(test_auc) and len(val_auc) > 0:
					select_Idx = np.argmax(val_auc)
					print_red(model_name)
					print_yellow('AUC: Best Test {0:.4f},  Val {1:.4f}'.format(test_auc[select_Idx], val_auc[select_Idx]))

print_green('The naviely-trained target model profermance')
target_folder = os.path.join(result_folder, 'FDA')
target_model_folders = glob.glob(target_folder +'/*')
present_auc(target_model_folders)