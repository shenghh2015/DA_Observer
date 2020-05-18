import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
from sklearn.decomposition import PCA

from load_data import *
from model import *

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

docker = True
## domain adaptation model
if not docker:
	output_folder = './data'
else:
	output_folder = '/data/DA_Observer/CLB_FDA'
	figure_folder = '/data/results/figures'
	generate_folder(figure_folder)

# load FDA and CLB data
Xs, _, _, ys, _, _ = load_source(train = 1000, valid = 400, test = 400, sig_rate = 0.035)
Xs = np.random.RandomState(0).normal(Xs, 2)
Xt, _, Xt_test, yt, _, yt_test = load_target(dataset = 'total', train = 1000, valid = 400, test = 400)
Xs = (Xs-np.min(Xs))/(np.max(Xs)-np.min(Xs)); Xt = (Xt-np.min(Xt))/(np.max(Xt)-np.min(Xt))

# Use PCA analysis to get the n most significant components
n_components = 800
pca1 = PCA(n_components = n_components); pca2 = PCA(n_components = n_components)
Xs_input, Xt_input = Xs.reshape(Xs.shape[0], -1), Xt.reshape(Xt.shape[0], -1)
Xs_PCA = pca1.fit_transform(Xs_input); Xt_PCA = pca2.fit_transform(Xt_input)

# t-SNE fitting
tsne = TSNE()
Xs_embedded = tsne.fit_transform(Xs_PCA); Xt_embedded = tsne.fit_transform(Xt_PCA);
ys_list, yt_list = [], []
for i in range(len(ys)):
	if ys[i] ==0:
		ys_list.append('S:SA')
	else:
		ys_list.append('S:SP')

	if yt[i] ==0:
		yt_list.append('T:SA')
	else:
		yt_list.append('T:SP')

fig = plt.figure()
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)
sns.scatterplot(Xs_embedded[:,0], Xs_embedded[:,1], hue=ys_list, legend='full', palette=palette, marker= '+')
palette = sns.color_palette("dark", 2)
sns.scatterplot(Xt_embedded[:,0], Xt_embedded[:,1], hue=yt_list, legend='full', palette=palette, markers='o')
# save figure
plt.savefig(figure_folder+'/source_target_image.png', dpi = 100)

gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
source_folder = output_folder + '/CLB/cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k'
DA_folder = 'experiments/total_mmd_tf/mmd-0.8-lr-0.0001-bz-400-iter-50000-scr-True-shar-True-fc-128-bn-False-tclf-0.0-sclf-1.0-tlabels-0-vclf-1-total-val-100'
source_model = source_folder + '/source-best'
DA_model = DA_folder + '/target_best'
target_folder = output_folder+'/experiments/total_mmd_tf/cnn-4-bn-True-trn-85000-bz-400-lr-1e-05-Adam-5.0k'
target_model = target_folder + '/source-best'

tf.keras.backend.clear_session()
# create a graph
x = tf.placeholder("float", shape=[None, 109,109, 1])
y = tf.placeholder("float", shape=[None, 1])

conv_net_src, h_src, source_logit = conv_classifier2(x, nb_cnn = 4, fc_layers = [128,1], bn = False, scope_name = 'source')
source_vars_list = tf.trainable_variables('source')
source_key_list = [v.name[:-2].replace('source', 'base') for v in tf.trainable_variables('source')]
source_key_direct = {}
for key, var in zip(source_key_list, source_vars_list):
	source_key_direct[key] = var
source_saver = tf.train.Saver(source_key_direct, max_to_keep=1)

with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	source_saver.restore(sess, source_model)
	SO_features = h_src.eval(session =sess, feed_dict = {x: np.expand_dims(Xt_test, axis = 3)})
	SO_logits = source_logit.eval(session =sess, feed_dict = {x: np.expand_dims(Xt_test, axis = 3)})
	SO_auc = roc_auc_score(yt_test, SO_logits)

tf.keras.backend.clear_session()
# create a graph
x = tf.placeholder("float", shape=[None, 109,109, 1])
y = tf.placeholder("float", shape=[None, 1])

conv_net_src, h_src, source_logit = conv_classifier(x, nb_cnn = 4, fc_layers = [128,1], bn = False, scope_name = 'source')
source_vars_list = tf.trainable_variables('source')
source_key_list = [v.name[:-2].replace('source', 'base') for v in tf.trainable_variables('source')]
source_key_direct = {}
for key, var in zip(source_key_list, source_vars_list):
	source_key_direct[key] = var
source_saver = tf.train.Saver(source_key_direct, max_to_keep=1)

with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	source_saver.restore(sess, DA_model)
	DA_features = h_src.eval(session = sess, feed_dict = {x: np.expand_dims(Xt_test, axis = 3)})
	DA_logits = source_logit.eval(session =sess, feed_dict = {x: np.expand_dims(Xt_test, axis = 3)})
	DA_auc = roc_auc_score(yt_test, DA_logits)

tf.keras.backend.clear_session()
# create a graph
x = tf.placeholder("float", shape=[None, 109,109, 1])
y = tf.placeholder("float", shape=[None, 1])

conv_net_src, h_src, source_logit = conv_classifier(x, nb_cnn = 4, fc_layers = [128,1], bn = True, scope_name = 'source')
source_vars_list = tf.trainable_variables('source')
source_key_list = [v.name[:-2].replace('source', 'base') for v in tf.trainable_variables('source')]
source_key_direct = {}
for key, var in zip(source_key_list, source_vars_list):
	source_key_direct[key] = var
source_saver = tf.train.Saver(source_key_direct, max_to_keep=1)

with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	source_saver.restore(sess, target_model)
	target_features = h_src.eval(session = sess, feed_dict = {x: np.expand_dims(Xt_test, axis = 3)})
	target_logits = source_logit.eval(session =sess, feed_dict = {x: np.expand_dims(Xt_test, axis = 3)})
	target_auc = roc_auc_score(yt_test, target_logits)

print('AUC: SO {0:.4f} DA {1:.4f} TO {2:.4f}'.format(SO_auc, DA_auc, target_auc))

tsne = TSNE()
SO_embedded = tsne.fit_transform(SO_features); DA_embedded = tsne.fit_transform(DA_features); TO_embedded = tsne.fit_transform(target_features)
y_SO_list, y_DA_list, y_TO_list = [], [], []
for i in range(len(yt_test)):
	if yt_test[i] ==0:
		y_SO_list.append('SO:SA')
		y_DA_list.append('DA:SA')
		y_TO_list.append('TO:SA')
	else:
		y_SO_list.append('SO:SP')
		y_DA_list.append('DA:SP')
		y_TO_list.append('TO:SP')

plt.clf()
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)
sns.scatterplot(SO_embedded[:,0], SO_embedded[:,1], hue=y_SO_list, legend='full', palette=palette)
plt.savefig(figure_folder+'/SO.png')
# palette = sns.color_palette("dark", 2)
# sns.scatterplot(DA_embedded[:,0], DA_embedded[:,1], hue=y_DA_list, legend='full', palette=palette, markers='s')

plt.clf()
sns.set(rc={'figure.figsize':(11.7,8.27)})
# palette = sns.color_palette("bright", 2)
# sns.scatterplot(SO_embedded[:,0], SO_embedded[:,1], hue=y_SO_list, legend='full', palette=palette, marker= '+')
palette = sns.color_palette("bright", 2)
sns.scatterplot(DA_embedded[:,0], DA_embedded[:,1], hue=y_DA_list, legend='full', palette=palette)
plt.savefig(figure_folder+'/DA.png')

plt.clf()
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)
sns.scatterplot(TO_embedded[:,0], TO_embedded[:,1], hue=y_TO_list, legend='full', palette=palette, markers='s')
plt.savefig(figure_folder+'/TO.png')

from helper_function import plot_feature_dist, plot_feature_pair_dist
plot_feature_dist(figure_folder+'/SO.png', SO_features[:400,:], SO_features[400:800,:])
plot_feature_dist(figure_folder+'/DA.png', DA_features[:400,:], DA_features[400:800,:])
plot_feature_dist(figure_folder+'/TO.png', target_features[:400,:], TO_features[400:800,:])

plot_feature_pair_dist(figure_folder+'/DA_TO.png', DA_features, target_features, yt_test, yt_test, label = ['DA', 'TO'])

