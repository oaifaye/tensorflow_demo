'''
Created on 2017年12月28日

@author: Administrator
'''
import tensorflow as tf 
import numpy as np 

x = tf.placeholder(tf.float32)
y=tf.sigmoid(x)  
# tf.initialize_all_variables()
with tf.Session() as sess:  
    print(sess.run(y,feed_dict={x:[[0,0],[0,0]]}))