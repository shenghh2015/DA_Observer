import tensorflow as tf
import numpy as np
import argparse
import os
from sklearn.metrics import roc_auc_score
import scipy.io
import math
import multiprocessing as mp
import time

pi = math.pi

f = scipy.io.loadmat('signal_K_new3.mat')
sig = f['signal']

parser = argparse.ArgumentParser()
parser.add_argument('ID',type=int)
parser.add_argument('numFilter',type = int)

parser.add_argument('batch_size',type = int)
parser.add_argument('optimizer',type=str)
parser.add_argument('learning_rate',type=float)

args = parser.parse_args()
ID = args.ID
numFilter = args.numFilter

batch_size = args.batch_size
optimizer = args.optimizer
learning_rate = args.learning_rate
#lambda0 = args.lambda0
#direct = args.dir
direct = 'output/ID'+str(ID)+'_structure1x7_numFiler'+str(numFilter)+'_batch'+str(batch_size)+'Optimizer_'+optimizer+'_learningRate_'+str(learning_rate)
os.mkdir(direct+'/')

direct_st = direct+'/statistics'
os.mkdir(direct_st+'/')

train_size = 100000

#Train_1 = 'H1_test_K_new_b3.dat'
Train_0 = 'b_source.dat'
#h1_train = np.fromfile(Train_1,dtype=np.float32,count = -1)
h0_train = np.fromfile(Train_0,dtype=np.float32,count = -1)
#h1_train = np.reshape(h1_train,(-1,64,64))
h0_train = np.reshape(h0_train,(-1,64,64))

b_test = h0_train[train_size:train_size+100,:,:] 
b_val = h0_train[train_size+100:train_size+200,:,:] 


test_size = 100

h1_test = np.random.normal(b_test + sig,20)
h0_test = np.random.normal(b_test,20)
test_data = np.concatenate((h1_test,h0_test),axis=0)


h1_val = np.random.normal(b_val + sig,20)
h0_val = np.random.normal(b_val,20)
val_data = np.concatenate((h1_val,h0_val),axis=0)

h0_train = h0_train[0:train_size,:,:]

x = tf.placeholder(tf.float32, shape=[None,64,64])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_flat = tf.reshape(x,[-1,64*64])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)
def conv2d(x,W):
  return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'SAME')

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
# First layer
W_conv11 = weight_variable([5,5,1,numFilter])
b_conv11 = bias_variable([numFilter])
x_image = tf.reshape(x,[-1,64,64,1])
h_conv11 = lrelu(conv2d(x_image, W_conv11) + b_conv11 )

W_conv12 = weight_variable([5,5,numFilter,numFilter])
b_conv12 = bias_variable([numFilter])
h_conv12 = lrelu(conv2d(h_conv11, W_conv12)+b_conv12)

W_conv13 = weight_variable([5,5,numFilter,numFilter])
b_conv13 = bias_variable([numFilter])
h_conv13 = lrelu(conv2d(h_conv12, W_conv13)+b_conv13)

W_conv14 = weight_variable([5,5,numFilter,numFilter])
b_conv14 = bias_variable([numFilter])
h_conv14 = lrelu(conv2d(h_conv13, W_conv14)+b_conv14)

W_conv15 = weight_variable([5,5,numFilter,numFilter])
b_conv15 = bias_variable([numFilter])
h_conv15 = lrelu(conv2d(h_conv14, W_conv15)+b_conv15)

W_conv16 = weight_variable([5,5,numFilter,numFilter])
b_conv16 = bias_variable([numFilter])
h_conv16 = lrelu(conv2d(h_conv15, W_conv16)+b_conv16)

W_conv17 = weight_variable([5,5,numFilter,numFilter])
b_conv17 = bias_variable([numFilter])
h_conv17 = lrelu(conv2d(h_conv16, W_conv17)+b_conv17)

h_pool1 = max_pool_2x2(h_conv17)

# Dense layer
W_fc1 = bias_variable([32*32*numFilter, 1])
b_fc1 = bias_variable([1])

h_pool1_flat = tf.reshape(h_pool1, [-1,32*32*numFilter])
y_conv = tf.matmul(h_pool1_flat, W_fc1) + b_fc1

saver = tf.train.Saver(max_to_keep=None)

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_, logits = y_conv))

# Optimizer
if optimizer=="Adam":
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
else:
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(1e-4,0.9,use_nesterov=True).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_y1 = np.ones([batch_size,1])
batch_y0 = np.zeros([batch_size,1])
batch_y = np.concatenate((batch_y1,batch_y0),axis=0)
test_y1 = np.ones([test_size,1])
test_y0 = np.zeros([test_size,1])
test_y = np.concatenate((test_y1,test_y0),axis=0)


#tt1 = np.zeros([test_size,1])
#tt0 = np.zeros([test_size,1])
train_loss = np.zeros([0,0])
train_auc = np.zeros([0,0])
train_accuracy = np.zeros([0,0])
test_loss =  np.zeros([0,0])
test_auc =  np.zeros([0,0])
test_accuracy = np.zeros([0,0])

val_loss =  np.zeros([0,0])
val_auc =  np.zeros([0,0])
time_duration = np.zeros([1,1])


tmp1 = np.zeros([batch_size,64,64])
tmp0 = np.zeros([batch_size,64,64])
sig = np.reshape(sig,[1,64,64])

start_time = time.time()
for i in range(10000000):
  #start_time = time.time()
  ii = int(i%(train_size/batch_size))
  if ii ==0:
     shuff = np.random.permutation(train_size)
  shuff_batch = shuff[ii*batch_size:(1+ii)*batch_size]
  tmp0 = h0_train[shuff_batch,:,:]
  tmp1 = tmp0 + sig  

  batch_x = np.concatenate((tmp1,tmp0),axis=0)
  batch_x = np.random.normal(batch_x,20)
  #train_logit = y_conv.eval(feed_dict={x:batch_x})
  #train_loss = np.append(train_loss,cross_entropy.eval(feed_dict={y_conv:train_logit, y_:batch_y}))
  sess.run(train_step,feed_dict={x:batch_x,y_:batch_y})

  if i%100 == 0:

    train_logit = y_conv.eval(session=sess,feed_dict={x:batch_x})
    train_loss = np.append(train_loss,cross_entropy.eval(session=sess,feed_dict={y_conv:train_logit, y_:batch_y}))
    train_stat = np.exp(train_logit)
    train_auc = np.append(train_auc,roc_auc_score(batch_y, train_stat))

    test_logit = y_conv.eval(session=sess,feed_dict={x:test_data})
    test_loss = np.append(test_loss,cross_entropy.eval(session=sess,feed_dict={y_conv:test_logit, y_:test_y}))
    test_stat = np.exp(test_logit)
    test_auc = np.append(test_auc,roc_auc_score(test_y, test_stat))


    val_logit = y_conv.eval(session=sess,feed_dict={x:val_data})
    val_loss = np.append(val_loss,cross_entropy.eval(session=sess,feed_dict={y_conv:val_logit, y_:test_y}))
    val_stat = np.exp(val_logit)
    val_auc = np.append(val_auc,roc_auc_score(test_y, val_stat))


    print("step %d, train auc %g val auc %g train loss %g val loss %g"%(i,train_auc[-1],val_auc[-1],train_loss[-1],val_loss[-1]))
    direct_m = direct+'/model_'+str(i)
    os.mkdir(direct_m+'/')
    save_path = saver.save(sess,direct_m+'/model.ckpt')
    print("model saved")
    #print(train_auc)

    #W_filters = W_conv1.eval()
    #W_filters_f = W_filters[:,:,0,:]
    #W_filters_f.tofile(direct+'/filters.dat')
    np.savetxt(direct+'/training_auc.txt',train_auc)
    np.savetxt(direct+'/testing_auc.txt',test_auc)
    np.savetxt(direct+'/training_loss.txt',train_loss)
    np.savetxt(direct+'/testing_loss.txt',test_loss)

    np.savetxt(direct+'/val_loss.txt',val_loss)
    np.savetxt(direct+'/val_auc.txt',val_auc)
    #np.savetxt(direct+'/training_accuracy.txt',train_accuracy)
    #np.savetxt(direct+'/testing_accuracy.txt', test_accuracy)
    np.savetxt(direct_st+'/statistics_'+str(i)+'.txt',test_stat)
    #np.savetxt(direct+'/statistics0.txt',tt0)
    time_duration = np.append(time_duration,time.time() - start_time)
    print("total time:%g batch time:%g"%(time_duration[-1],time_duration[-1]-time_duration[-2]))
    np.savetxt(direct+'/time_duration.txt',time_duration)





