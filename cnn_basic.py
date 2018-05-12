# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 10:06:50 2018

@author: User
"""

""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from PIL import Image


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("./MNIST_DATA/", one_hot=True)

# Define paramaters for the model
learning_rate = 0.01
batch_size = 100
n_epochs = 10
n_train = 60000
n_test = 10000

# Step 1: Read in data



# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
#with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):    
#W1 = tf.get_variable("W1",shape=[4,4,1,8])
#W2 = tf.get_variable("W2",shape=[2,2,8,16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

#x_image= tf.reshape(x,[-1,28,28,1])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


W1= weight_variable([4,4,1,8])
W2= weight_variable([2,2,8,8])
W3=weight_variable([4,4,8,16])
W4=weight_variable([2,2,16,16])
    
def neural_network_model(data):
    
    Z1=tf.nn.conv2d(data,W1,strides=[1,1,1,1], padding='SAME' )
    A1=tf.nn.relu(Z1)
    Z2=tf.nn.conv2d(A1,W2,strides=[1,1,1,1], padding='SAME' )
    A2=tf.nn.relu(Z2)
    P1=tf.nn.max_pool(A2,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    Z3=tf.nn.conv2d(P1,W3,strides=[1,2,2,1], padding='SAME')
    A3=tf.nn.relu(Z3)
    Z4=tf.nn.conv2d(A3,W4,strides=[1,1,1,1], padding='SAME')
    A4=tf.nn.relu(Z4)
    P2=tf.nn.max_pool(A4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    P2=tf.contrib.layers.flatten(P2)
    output=tf.contrib.layers.fully_connected(P2, 10, activation_fn=None)
    return output
# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer

'''def test_image():
    fname = "digit5.png"
    image = np.array(ndimage.imread(fname, flatten=True, mode='L'))
    my_image = scipy.misc.imresize(image, size=(28,28,1))
    #plt.imshow(my_image)
    #image=Image.open(fname).convert('LA')
    return my_image
'''


'''
x_image=tf.reshape(x,[-1,28,28,1])
Z3=neural_network_model(x_image)
preds=tf.nn.softmax(Z3)
predict=tf.argmax(preds,1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=Z3))
optimizer = tf.train.AdamOptimizer().minimize(loss)
correct=tf.equal(tf.argmax(preds,1), tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct,'float32'))
'''    
    
def train_neural_network(x):
    x_image=tf.reshape(x,[-1,28,28,1])
    Z3=neural_network_model(x_image)
    preds=tf.nn.softmax(Z3)
    predict=tf.argmax(preds,1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=Z3))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    correct=tf.equal(tf.argmax(preds,1), tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct,'float32'))    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c= sess.run([optimizer,loss], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss+=c
            print('Epoch:', epoch, 'completed out of:',n_epochs,'loss:',epoch_loss)
            train_accuracy = accuracy.eval(feed_dict={x: epoch_x, y: epoch_y})
            print('step %d, training accuracy %g' % (epoch, train_accuracy))
        
        #print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        #my_image=test_image()
        #save_path = saver.save(sess, "/tmp/model.ckpt")
        #predict_op = tf.argmax(preds, 1)
        #saver=tf.train.Saver()
        tf.add_to_collection("predict", predict)
        #saver.save(sess, "./my_model")
        '''predictions= predict.eval({x:mnist.test.images})
        print('label:', mnist.test.labels[100])
        print(predictions[100])
        two_d = (np.reshape(mnist.test.images[100], (28, 28)) * 255).astype(np.uint8)
        plt.imshow(two_d, interpolation='nearest')
'''
train_neural_network(x)

#def test_model():
with tf.Session() as sess:
        #new_saver=tf.train.import_meta_graph("./my_model.meta")
        #new_saver.restore(sess,'./my_model')
        predict_op1 = tf.get_collection("predict")
        #saver.restore(sess, "/tmp/model.ckpt")
        #sess.run(tf.global_variables_initializer())
        W5=W1.eval()
        W6=W2.eval()
        W7=W3.eval()
        W8=W4.eval()
        #W9=tf.get_variable('fully_connected_1/weights')
        #sess.run(tf.global_variables_initializer())
        #predictions=predict.eval(feed_dict={x:mnist.test.images, W1:W5, W2:W6,W3:W7, W4:W8 })
        predictions=sess.run(predict_op1,feed_dict={x:mnist.test.images})
        print(predictions[50])
        print('label:', mnist.test.labels[50])
#test_model()


        

# Step 7: calculate accuracy with test set
'''preds = tf.nn.softmax(Z3)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

#writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer()))
   for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        optimizer.run(feed_dict={x: batch[0], y: batch[1]})

        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
'''