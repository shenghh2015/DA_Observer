import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
from sklearn.decomposition import PCA

def plot_LOSS(file_name, train_loss_list, val_loss_list, test_loss_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	ax = fig.add_subplot(111)
	ax.plot(train_loss_list)
	ax.plot(val_loss_list)
	ax.plot(test_loss_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Loss')
	ax.legend(['D','S','T'])
	ax.set_xlim([0,len(train_loss_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_AUCs(file_name, train_list, val_list, test_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
	ax.plot(train_list)
	ax.plot(val_list)
	ax.plot(test_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC')
	ax.legend(['Train','Valid','Test'])
	ax.set_xlim([0,len(train_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_src_trg_AUCs(file_name, train_list, val_list, test_list, src_test_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
	ax.plot(train_list)
	ax.plot(val_list)
	ax.plot(test_list)
	ax.plot(src_test_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC')
	ax.legend(['T-Train','T-Valid','T-Test', 'S-Test'])
	ax.set_xlim([0,len(train_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_AUCs_DomACC(file_name, train_list, val_list, test_list, dom_acc_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
	ax.plot(train_list)
	ax.plot(val_list)
	ax.plot(test_list)
	ax.plot(dom_acc_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC/ACC')
	ax.legend(['AUC:Train','AUC:Valid','AUC:Test','ACC:Dom'])
	ax.set_xlim([0,len(train_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost

# plot and save the file
def plot_loss(model_name, loss, val_loss, file_name):
	generate_folder(model_name)
	f_out = file_name
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	start_idx = 0
	if len(loss)>start_idx:
		title = os.path.basename(os.path.dirname(file_name))
		fig = Figure(figsize=(8,6))
		ax = fig.add_subplot(1,1,1)
		ax.plot(loss[start_idx:],'b-',linewidth=1.3)
		ax.plot(val_loss[start_idx:],'r-',linewidth=1.3)
		ax.set_title(title)
		ax.set_ylabel('Loss')
		ax.set_xlabel('batches')
		ax.legend(['D-loss', 'G-loss'], loc='upper left')  
		canvas = FigureCanvasAgg(fig)
		canvas.print_figure(f_out, dpi=80)

def plot_auc_iterations(target_auc_list, val_auc_list, target_file_name):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = target_file_name
	ax = fig.add_subplot(111)
	ax.plot(target_auc_list)
	ax.plot(val_auc_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC')
	ax.legend(['Test','Val'])
	ax.set_xlim([0,len(target_auc_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_gradients(file_name, dis_grad_list1, dis_grad_list2, gen_grad_list1, gen_grad_list2):
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (10,9)
	fig = Figure(figsize=fig_size)
	file_name = target_file_name
	ax = fig.add_subplot(441)
	ax.plot(dis_grad_list1)
	ax = fig.add_subplot(442)
	ax.plot(dis_grad_list2)
	ax = fig.add_subplot(443)
	ax.plot(gen_grad_list1)
	ax = fig.add_subplot(444)
	ax.plot(gen_grad_list2)
# 	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC')
	ax.legend(['Test','Val'])
	ax.set_xlim([0,len(target_auc_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_auc_dom_acc_iterations(target_auc_list, val_auc_list, dom_acc_list, target_file_name):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = target_file_name
	ax = fig.add_subplot(111)
	ax.plot(target_auc_list)
	ax.plot(val_auc_list)
	ax.plot(dom_acc_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC/ACC')
	ax.legend(['AUC:Test','AUC:Val', 'ACC:Dom'])
	ax.set_xlim([0,len(target_auc_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)


def plot_src_trg_auc_iterations(target_auc_list, val_auc_list, src_auc_list, target_file_name):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = target_file_name
	ax = fig.add_subplot(111)
	ax.plot(target_auc_list)
	ax.plot(val_auc_list)
	ax.plot(src_auc_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC/ACC')
	ax.legend(['T-Test','T-Val', 'S-Test'])
	ax.set_xlim([0,len(target_auc_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

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

def plot_feature_dist(file_name, feat, y, method_label):
	import matplotlib.pyplot as plt
	fig_size = (8,6)
	fig = plt.figure(figsize=fig_size)
	tsne = TSNE(random_state=0)
	feat = np.squeeze(feat); y = y.flatten()
	embedded = tsne.fit_transform(feat)
	y_list = []
	for i in range(len(y)):
		if y[i] == 0:
			y_list.append('{}:SA'.format(method_label))
		else:
			y_list.append('{}:SP'.format(method_label))
	sns.set(rc={'figure.figsize':(11.7,8.27)})
	palette = sns.color_palette("bright", 2)
	sns.scatterplot(embedded[:,0], embedded[:,1], hue=y_list, legend='full', palette=palette)
	plt.savefig(file_name, dpi=100)
	plt.close('all')

def plot_feature_pair_dist(file_name, source_feat, target_feat, source_y, target_y, label = ['source', 'target']):
	import matplotlib.pyplot as plt
	fig_size = (8,6)
	fig = plt.figure(figsize=fig_size)
	sns.set(rc={'figure.figsize':(11.7,8.27)})
	tsne = TSNE(random_state=0)
	source_feat, target_feat = np.squeeze(source_feat), np.squeeze(target_feat)
	source_embedded = tsne.fit_transform(source_feat); target_embedded = tsne.fit_transform(target_feat)
	source_y, target_y = source_y.flatten(), target_y.flatten()
	source_y_list, target_y_list = [], []
	for i in range(len(source_y)):
		if source_y[i] == 0:
			source_y_list.append('{}:SA'.format(label[0]))
		else:
			source_y_list.append('{}:SP'.format(label[0]))
	for i in range(len(target_y)):
		if target_y[i] == 0:
			target_y_list.append('{}:SA'.format(label[1]))
		else:
			target_y_list.append('{}:SP'.format(label[1]))
	palette = sns.color_palette("bright", 2)
	sns.scatterplot(source_embedded[:,0], source_embedded[:,1], hue=source_y_list, legend='brief', palette=palette, marker = 'o')
	palette = sns.color_palette("dark", 2)
	sns.scatterplot(target_embedded[:,0], target_embedded[:,1], hue=target_y_list, legend='brief', palette=palette, marker = 'X')
	plt.savefig(file_name, dpi=100)
	plt.close('all')
	
	
	
	
	
	
	
	
	
