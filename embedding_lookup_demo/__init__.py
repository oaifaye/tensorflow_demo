# -*- coding:utf-8 -*-  
#功能： 使用tensorflow实现一个简单的逻辑回归  
import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  

train_inputs = tf.placeholder(tf.int32)
embedding_dict = tf.Variable(tf.random_uniform([10,5],-1.0,1.0))
# embedding_dict = tf.placeholder(tf.int32)
embed = tf.nn.embedding_lookup(embedding_dict, train_inputs) # batch_size

# vec_l2_model = tf.sqrt(  # 求各词向量的L2模
#                 tf.reduce_sum(tf.square(train_inputs),1,keep_dims=True))

with tf.Session() as sess:  
    #初始化所有变量  
    sess.run(tf.initialize_all_variables())  
    print('embedding_dict:',sess.run(embedding_dict))
    train_inputs1 = [[0,1,2],[0,1,3],[0,1,3],[0,1,3],[0,1,3]]
    print('embed:',sess.run(embed,feed_dict={train_inputs:train_inputs1}))
