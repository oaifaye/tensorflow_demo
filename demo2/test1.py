'''
Created on 2017年12月26日

@author: Administrator
'''
# -*- coding: utf-8 -*-

import tensorflow as tf 

a = tf.constant([1,2])
b = tf.constant([0,2])
c = tf.subtract(a, b)
sess = tf.Session()
print (sess.run(c))
sess.close()