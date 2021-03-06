# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 10:06:50 2018

@author: User
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


#Building the model and returning the 10x1 data

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
  

#Training the model
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
        #saver.save(sess, "./my_model")
        #predict_op = tf.argmax(preds, 1)
        
        #Saving the model.
        saver=tf.train.Saver()
        tf.add_to_collection("predict", predict)
        saver.save(sess, "/tmp/model.ckpt")
        
        
        #predictions= predict.eval({x:mnist.test.images})
        #print('label:', mnist.test.labels[100])
        #print(predictions[100])
        #two_d = (np.reshape(mnist.test.images[100], (28, 28)) * 255).astype(np.uint8)
        #plt.imshow(two_d, interpolation='nearest')
        
train_neural_network(x)

#def test_model():

#Here, I have wanted to use trained parameters to evaluate the test set.
#But it is not working
with tf.Session() as sess:
        #new_saver=tf.train.import_meta_graph("./my_model.meta")
        #new_saver.restore(sess,'./my_model')
        predict_op1 = tf.get_collection("predict")
        saver.restore(sess, "/tmp/model.ckpt")
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


        

