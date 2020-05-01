import os
import numpy as np
import glob
from termcolor import colored 

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

result_folder = '/data/backup'
method = 'adda'
dataset = 'dense'

## source model training
print_green('The source model profermance')
source_folder = os.path.join(result_folder, 'CLB')
source_model_folders = glob.glob(source_folder +'/*')
for source_model in source_model_folders:
	if os.path.isdir(source_model) and os.path.exists(source_model+'/source-best.meta'):
		print_red(source_model)
		#load the AUC performance
		val_auc = np.loadtxt(source_model+'/val_auc.txt')
		test_auc = np.loadtxt(source_model+'/testing_auc.txt')
		select_Idx = np.argmax(val_auc)
		print_yellow('AUC: Best Val {0:.4f}, Test {1:.4f}'.format(val_auc[select_Idx], test_auc[select_Idx]))

## directly train the model


## transfer learning

## ADDA

## mmd

## ADDA + target labels

## mmd + target labels
