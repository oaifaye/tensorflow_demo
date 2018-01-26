'''
Created on 2018年1月13日

@author: Administrator
tf非线性回归
http://blog.csdn.net/y12345678904/article/details/77743696
'''

import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  
  
  
#生成随机变量  
# x_data = np.linspace(-1, 1, 12)#生成200个随机点，范围为-1 --> 1  
# x_data = x_data.reshape((12, 1))#维度设置为(200, 1)  
# noise = np.random.normal(0, 0.02, x_data.shape)#生成干扰项  
#     
# y_data = np.square(x_data) + noise  
  
# x_data = [[0.01*8],[0.02*8],[0.03*8],[0.04*8],[0.05*8],[0.06*8],[0.07*8],[0.08*8],[0.09*8],[0.10*8],[0.11*8],[0.12*8]]#维度设置为(200, 1)  
x_data = [[-0.16*6],[-0.16*5],[-0.16*4],[-0.16*3],[-0.16*2],[-0.16*1],[0.16*1],[0.16*2],[0.16*3],[0.16*4],[0.16*5],[0.16*6]]#维度设置为(200, 1)  
# x_data = [[0.21902],[0.24511],[0.26545],[0.27891],[0.30306],[0.31884],[0.15473],[0.16705],[0.18754],[0.19872],[0.20558],[0.20169]]
y_data = [[0.28643],[0.31321],[0.34662],[0.37498],[0.39484],[0.35561],[0.22857],[0.25087],[0.28978],[0.30504],[0.29461],[0.27542]]
  
# x_data = [[-0.99],[-0.85],[-0.75],[-0.55],[-0.5],[-0.25],[0.00],[0.25],[0.5],[0.55],[0.75],[1.00]]
# y_data = [[0.8],[0.5],[0.18],[0.24],[0.45],[0.6],[0.8],[0.85],[0.6],[0.4],[0.4],[0.2]]
  
x = tf.placeholder(tf.float32, [None, 1])  
y = tf.placeholder(tf.float32, [None, 1])  
  
#定义神经网络中间层权值  
weights_l1 = tf.Variable(tf.random_normal([1, 10]))#10个神经元  
biases_l1 = tf.Variable(tf.zeros([1, 10]))  
wx_plust_b_l1 = tf.matmul(x, weights_l1) + biases_l1  
# l1 = tf.nn.relu(wx_plust_b_l1)#双曲正切函数作为激活函数  
l1 = tf.nn.tanh(wx_plust_b_l1)#双曲正切函数作为激活函数  
# l1 = tf.sigmoid(wx_plust_b_l1)
  
#定义输出层  
weights_l2 = tf.Variable(tf.random_normal([10, 2]))#输出层1个神经元  
biases_l2 = tf.Variable(tf.zeros([1, 1]))#一个偏置  
wx_plust_b_l2 = tf.matmul(l1, weights_l2) + biases_l2  
# prediction = tf.nn.relu(wx_plust_b_l2)#预测结果  
prediction = tf.nn.tanh(wx_plust_b_l2)#预测结果  
# prediction = tf.sigmoid(wx_plust_b_l2)#预测结果  
  
#代价函数  
loss = tf.reduce_mean(tf.square(y - prediction))  
#使用梯度下降  
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#学习率设置为0.1  
  
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())#变量初始化，一定要做  
    for _ in range(200000):  
        
        sess.run(train_step, feed_dict={x:x_data, y:y_data})#使用梯度下降法进行训练参数  
    print('weights_l1:',sess.run(weights_l1))
    print('biases_l1:',sess.run(biases_l1))
    #获得预测值  
    prediction_value = sess.run(prediction, feed_dict={x: x_data})#得到预测结果  
    print(sess.run(prediction, feed_dict={x: [[0.16*6]]}))
    #画图  
    plt.figure()  
    plt.scatter(x_data, y_data)#画散点图  
    plt.plot(x_data, prediction_value, 'r-', lw = 5)#画预测的实线，红色  
    plt.show()
