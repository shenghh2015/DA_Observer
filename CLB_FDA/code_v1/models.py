from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input, Cropping2D, concatenate, Add, LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D, Conv2D,Lambda, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, ELU  
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import keras.backend as K
import keras

def _conv_relu(nb_filter, kernel_size, kernel_initializer='truncated_normal', bn = False):
	def f(input):
		conv_a = Conv2D(nb_filter, kernel_size, padding='same', 
			kernel_initializer= kernel_initializer, bias_initializer='zeros')(input)
		if bn:
			norm_a = BatchNormalization()(conv_a)
		else:
			norm_a = conv_a
		norm_a = LeakyReLU(alpha = 0.2)(norm_a)
		return norm_a
	return f

def cnn_block_base(input, nb_cnn = 7, kernel_size = (5,5), nb_filter = 32, kernel_initializer='truncated_normal', bn = False):
	cnn_blk = _conv_relu(nb_filter, kernel_size, kernel_initializer, bn)(input)
	for i in range(nb_cnn-1):
		cnn_blk = _conv_relu(nb_filter,kernel_size, kernel_initializer, bn)(cnn_blk)
	pool = MaxPooling2D(pool_size = (2,2))(cnn_blk)
	return pool

## resNet-like
def _conv_bn_relux2_v2(nb_filter, kernel_size, kernel_initializer = 'orthogonal', bn = True):
	def f(input):
		conv_a = Conv2D(nb_filter, kernel_size, padding='same', 
			kernel_initializer=kernel_initializer, kernel_regularizer = l2(weight_decay))(input)
		if bn:
			conv_a = BatchNormalization()(conv_a)
		act_a = LeakyReLU(alpha = 0.2)(conv_a)
		conv_b = Conv2D(nb_filter, kernel_size, padding='same', 
			kernel_initializer=kernel_initializer, kernel_regularizer = l2(weight_decay))(act_a)
		if bn:
			conv_b = BatchNormalization()(conv_b)
		act_b = LeakyReLU(alpha = 0.2)(conv_b)
		return act_b
	return f

def _res_conv_bn_relu_v2(nb_filter, kernel_size, kernel_initializer, bn):
	def f(input):
		conv_ = _conv_bn_relux2_v2(nb_filter, kernel_size, kernel_initializer, bn)(input)
		add_ = keras.layers.Add()([input, conv_])
		return add_
	return f

def classifier_base(input_):
	flat = Flatten()(input_)
	den = Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(flat)
	score = Activation('sigmoid')(den)
	return score

## for scoure model and target model trained directly
def buildCNNClassifierModel(input_shape = (109,109,1), kernel_size = (5,5), nb_filter = 32, kernel_initializer='truncated_normal', bn = False):
	input_ =Input(shape = input_shape)
	cnn_blk1 = cnn_block_base(input_, 2, kernel_size, nb_filter, kernel_initializer, bn)
	cnn_blk2 = cnn_block_base(cnn_blk1, 2, kernel_size, nb_filter, kernel_initializer, bn)
	output_ = classifier_base(cnn_blk2)
	model = Model(input = input_, output = output_)
	return model
 
def buildClassiferModel(input_shape):
	input_ = Input(shape = input_shape)
	out_ = classifier_base(input_)
	model = Model(input = input_, output = out_)	
	return model

### discriminator design v1
def discriminator_base_v1(input):
	nb_filter = 128
	fc1 = Dense(nb_filter)(input)
	fc2 = Dense(1)(fc1)
	out_ = Activation('sigmoid')(fc2)
	return out_

def buildDiscModel_v1(input_shape):
	input_ = Input(shape = input_shape)
	out_ = discriminator_base_v1(input_)
	model = Model(input = input_, output = out_)
	return model
