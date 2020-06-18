import numpy as np
import tensorflow as tf
from models2 import *

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def lerp(a, b, t):
    with tf.name_scope('Lerp'):
        return a + (b - a) * t

def lerp_clip(a, b, t):
    with tf.name_scope('LerpClip'):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


# def G_wgan_acgan(G, D, opt, training_set, minibatch_size,
#     cond_weight = 1.0): # Weight of the conditioning term.
# 
#     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     labels = training_set.get_random_labels_tf(minibatch_size)
#     fake_images_out = G.get_output_for(latents, labels, is_training=True)
#     fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
#     loss = -fake_scores_out
# 
#     if D.output_shapes[1][1] > 0:
#         with tf.name_scope('LabelPenalty'):
#             label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
#         loss += label_penalty_fakes * cond_weight
#     return loss

def D_wgangp_acgan(reals, fakes, minibatch_size, dis_training, dis_cnn = 4, fc_layers = [128, 1], dis_bn = True,
	wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
	wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
	wgan_target     = 1.0):      # Target value for gradient magnitudes.
	
	fake_scores_out = discriminator(fakes, nb_cnn = dis_cnn, fc_layers = fc_layers, bn = dis_bn, reuse = True, drop = 0, bn_training = dis_training)
	real_scores_out = discriminator(reals, nb_cnn = dis_cnn, fc_layers = fc_layers, bn = dis_bn, reuse = True, drop = 0, bn_training = dis_training)
	loss = tf.reduce_mean(fake_scores_out) - tf.reduce_mean(real_scores_out)
	
	mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fakes.dtype)
	mixed_input = lerp(tf.cast(reals, fakes.dtype), fakes, mixing_factors)
	mixed_scores = fp32(discriminator(mixed_input))
	mixed_loss = tf.reduce_sum(mixed_scores)
	mixed_grads = fp32(tf.gradients(mixed_loss, [mixed_input])[0])
	mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
	gradient_penalty = tf.square(mixed_norms - wgan_target)
	loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

	epsilon_penalty = tf.square(real_scores_out)
	loss += epsilon_penalty * wgan_epsilon

	return loss