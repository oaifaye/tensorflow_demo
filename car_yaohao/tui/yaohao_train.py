'''
Created on 2018年1月13日

@author: Administrator
tf多特征非线性回归
https://www.2cto.com/kf/201704/626628.html
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
# x_data = [[-0.16*6,0.1],[-0.16*5,0.2],[-0.16*4,0.3],[-0.16*3,0.4],[-0.16*2,0.5]]#维度设置为(200, 1)  
# y_data = [[0.28643],[0.31321],[0.34662],[0.37498],[0.39484]]
  
year_2014 = 0.1
year_2015 = 0.2
year_2016 = 0.3
year_2017 = 0.4
year_2018 = 0.5

seed_plt = 0.042

x_data = [
        [-seed_plt*23,0.11789,0.11910],[-seed_plt*22,0.12477,0.12820],
        [-seed_plt*21,0.14280,0.14806],[-seed_plt*20,0.15508,0.16252],[-seed_plt*19,0.16890,0.17658],
        [-seed_plt*18,0.18292,0.19115],[-seed_plt*17,0.19867,0.20520],[-seed_plt*16,0.19941,0.20397],
        [-seed_plt*15,0.16605,0.16883],[-seed_plt*14,0.15972,0.16275],[-seed_plt*13,0.16640,0.16952],
          
        [-seed_plt*12,0.15151,0.15679],[-seed_plt*11,0.15752,0.16554],[-seed_plt*10,0.18508,0.19272],
        [-seed_plt*9,0.19853,0.21006],[-seed_plt*8,0.21182,0.22585],[-seed_plt*7,0.32371,0.24690],
        [-seed_plt*6,0.25987,0.27367],[-seed_plt*5,0.28182,0.29401],[-seed_plt*4,0.28292,0.29100],
        [-seed_plt*3,0.22107,0.22590],[-seed_plt*2,0.22458,0.23236],[-seed_plt*1,0.23198,0.24061],
          
        [seed_plt*1,0.21902,0.23076],[seed_plt*2,0.24511,0.25761],[seed_plt*3,0.26545,0.28025],
        [seed_plt*4,0.27891,0.29496],[seed_plt*5,0.30306,0.32079],[seed_plt*6,0.31664,0.32712],
        [seed_plt*7,0.15473,0.15936],[seed_plt*8,0.16705,0.17397],[seed_plt*9,0.18764,0.19823],
        [seed_plt*10,0.19872,0.21006],[seed_plt*11,0.20558,0.21532],[seed_plt*12,0.20169,0.21244],
          
        ]

y_ave_data = [
#         [0.16340],[0.14262],[0.11891],
#         [0.11144],[0.10877],[0.10757],
#         [0.11162],[0.12950],[0.16892],
#         [0.26201],[0.33987],
          
#           [0.25861],
          [0.13729],[0.15485],
          [0.18996],[0.22682],[0.23539],
          [0.24490],[0.24537],[0.22391],
          [0.19093],[0.18588],[0.19599],
          
          [0.19207],[0.20103],[0.24344],
          [0.28478],[0.32225],[0.34766],
          [0.35522],[0.35356],[0.32422],
          [0.26656],[0.27150],[0.28916],
          
          [0.28643],[0.31321],[0.34662],
          [0.37498],[0.39484],[0.35561],
          [0.22857],[0.25087],[0.28978],
          [0.30504],[0.29461],[0.27542]
        ]

y_min_data = [
#         [0.10000],[0.10000],[0.10000],
#           [0.10000],[0.10000],[0.10000],
#           [0.10500],[0.12000],[0.16000],
#           [0.24400],[0.25000],
          
#           [0.10000],
          [0.11600],[0.14200],
          [0.18000],[0.20100],[0.21100],
          [0.22100],[0.20100],[0.15000],
          [0.15500],[0.17000],[0.18500],
          
          [0.18000],[0.19200],[0.23500],
          [0.27200],[0.30100],[0.31600],
          [0.32000],[0.30400],[0.20500],
          [0.23100],[0.25700],[0.27700],
          
          [0.27500],[0.30100],[0.33600],
          [0.36000],[0.34800],[0.25000],#xx
          [0.19000],[0.23500],[0.27200],
          [0.27000],[0.25500],[0.24000]
        ]
  
# 特征数
featurenum = 3
x = tf.placeholder(tf.float32, [None, featurenum])  
y = tf.placeholder(tf.float32, [None, 1]) 
  
#定义神经网络中间层权值  
weights_l1 = tf.Variable(tf.random_normal([featurenum, 10]))#10个神经元  
biases_l1 = tf.Variable(tf.zeros([1, 10]))  
wx_plust_b_l1 = tf.matmul(x, weights_l1) + biases_l1  
# l1 = tf.nn.relu(wx_plust_b_l1)#双曲正切函数作为激活函数  
l1 = tf.nn.tanh(wx_plust_b_l1)#双曲正切函数作为激活函数  
# l1 = tf.sigmoid(wx_plust_b_l1)
  

#定义输出层  
weights_l2 = tf.Variable(tf.random_normal([10, 1]))#输出层1个神经元  
biases_l2 = tf.Variable(tf.zeros([1,1]))#一个偏置  
wx_plust_b_l2 = tf.matmul(l1, weights_l2) + biases_l2  
# prediction = tf.nn.relu(wx_plust_b_l2)#预测结果  
prediction = tf.nn.tanh(wx_plust_b_l2)#预测结果  
# prediction = tf.sigmoid(wx_plust_b_l2)#预测结果  

tf.add_to_collection(tf.GraphKeys.WEIGHTS, biases_l1)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, biases_l2)
regularizer = tf.contrib.layers.l2_regularizer(scale=5.0/10000)
reg_term = tf.contrib.layers.apply_regularization(regularizer) 
  
#代价函数  
loss = tf.reduce_mean(tf.square(y - prediction))  +reg_term
#使用梯度下降  
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#学习率设置为0.1  
  
with tf.Session() as sess:  
    
#     run_mode = 'ave'
    run_mode = 'min'
    
    sess.run(tf.global_variables_initializer())#变量初始化，一定要做  
    if(run_mode == 'ave'):
        y_data = y_ave_data
        min_loss = 0.0002
        print_weights_l1_p = 'weights_ave_l1_p'
        print_biases_l1_p = 'biases_ave_l1_p'
        print_weights_l2_p = 'weights_ave_l2_p'
        print_biases_l2_p = 'biases_ave_l2_p'
    elif(run_mode == 'min'):
        y_data = y_min_data
        min_loss = 0.0002
        print_weights_l1_p = 'weights_min_l1_p'
        print_biases_l1_p = 'biases_min_l1_p'
        print_weights_l2_p = 'weights_min_l2_p'
        print_biases_l2_p = 'biases_min_l2_p'
        
    i = 0
    while True:  
        i =  i+1
        print(i)
        sess.run(train_step, feed_dict={x:x_data, y:y_data})#使用梯度下降法进行训练参数  
        lossi  = sess.run(loss, feed_dict={x:x_data, y:y_data})
        print('lossi:',lossi)
        if(lossi < min_loss):
            break;
    print(print_weights_l1_p ,'=',sess.run(weights_l1).tolist())
    print(print_biases_l1_p ,'=',sess.run(biases_l1).tolist())
    print(print_weights_l2_p,'=',sess.run(weights_l2).tolist())
    print(print_biases_l2_p,'=',sess.run(biases_l2).tolist())
    print('loss:',sess.run(loss, feed_dict={x:x_data, y:y_data}))
    
    #获得预测值  
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    print('预测1：',prediction_value)
    prediction_value1 = sess.run(prediction, feed_dict={x: [[seed_plt*13,0.19169,0.20244]]})
    print('预测2：',prediction_value1)
    
    #画图  
    seed_plt = 0.042
    x_plt = [[-seed_plt*23],[-seed_plt*22],[-seed_plt*21],[-seed_plt*20],[-seed_plt*19],[-seed_plt*18],[-seed_plt*17],[-seed_plt*16],[-seed_plt*15],[-seed_plt*14],
         [-seed_plt*13],[-seed_plt*12],[-seed_plt*11],[-seed_plt*10],[-seed_plt*9],[-seed_plt*8],[-seed_plt*7],[-seed_plt*6],[-seed_plt*5],[-seed_plt*4],
         [-seed_plt*3],[-seed_plt*2],[-seed_plt*1],[seed_plt*1],[seed_plt*2],[seed_plt*3],[seed_plt*4],[seed_plt*5],[seed_plt*6],[seed_plt*7],
         [seed_plt*8],[seed_plt*9],[seed_plt*10],[seed_plt*11],[seed_plt*12],
#          [seed_plt*13],[seed_plt*14],[seed_plt*15],[seed_plt*16],[seed_plt*17],[seed_plt*18],[seed_plt*19],[seed_plt*20],[seed_plt*21],[seed_plt*22],[seed_plt*23],[seed_plt*24]
         ]
    
    plt.figure()  
    prediction_plt1 = prediction_value.tolist()
    plt.scatter(x_plt, y_data)#画散点图  
#     plt.scatter(x_plt, prediction_plt1)#画预测的
    plt.plot(x_plt, prediction_plt1, 'r-', lw = 5)#画预测的实线，红色  
    prediction_plt = prediction_value.tolist()
    plt.show()
