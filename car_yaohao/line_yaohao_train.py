'''
Created on 2018年1月13日

@author: Administrator
tf多特征线性回归
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
  
year_2014 = 2014
year_2015 = 2015
year_2016 = 2016
year_2017 = 2017
year_2018 = 2018
x_data = [
#           [-0.16*5,year_2014,0.12582,0.13354],[-0.16*4,year_2014,0.12845,0.13115],
#           [-0.16*3,year_2014,0.11157,0.11281],[-0.16*2,year_2014,0.10621,0.10677],[-0.16*1,year_2014,0.10432,0.10466],
#           [0.16*1,year_2014,0.10363,0.10392],[0.16*2,year_2014,0.10425,0.10474],[0.16*3,year_2014,0.11104,0.11348],
#           [0.16*4,year_2014,0.13378,0.14019],[0.16*5,year_2014,0.18119,0.19272],[0.16*6,year_2014,0.22907,0.24303],
          
#           [-0.16*6,year_2015,0.22875,0.23774],
        [2,year_2015,11789,11910],[3,year_2015,12477,12820],
        [4,year_2015,14280,14806],[5,year_2015,15508,16252],[6,year_2015,16890,17658],
        [7,year_2015,18292,19115],[8,year_2015,19867,20520],[9,year_2015,19941,20397],
        [10,year_2015,16605,16883],[11,year_2015,15972,16275],[12,year_2015,16640,16952],
          
        [1,year_2016,15151,15679],[2,year_2016,15752,16554],[3,year_2016,18508,19272],
        [4,year_2016,19853,21006],[5,year_2016,21182,22585],[6,year_2016,32371,24690],
        [7,year_2016,25987,27367],[8,year_2016,28182,29401],[9,year_2016,28292,29100],
        [10,year_2016,22107,22590],[11,year_2016,22458,23236],[12,year_2016,23198,24061],
          
        [1,year_2017,21902,23076],[2,year_2017,24511,25761],[3,year_2017,26545,28025],
        [4,year_2017,27891,29496],[5,year_2017,30306,32079],[6,year_2017,31664,32712],
        [7,year_2017,15473,15936],[8,year_2017,16705,17397],[9,year_2017,18764,19823],
        [10,year_2017,19872,21006],[11,year_2017,20558,21532],[12,year_2017,20169,21244],
          
        ]

y_ave_data = [
#         [0.16340],[0.14262],[0.11891],
#         [0.11144],[0.10877],[0.10757],
#         [0.11162],[0.12950],[0.16892],
#         [0.26201],[0.33987],
          
#           [0.25861],
          [13729],[15485],
          [18996],[22682],[23539],
          [24490],[24537],[22391],
          [19093],[18588],[19599],
          
          [19207],[20103],[24344],
          [28478],[32225],[34766],
          [35522],[35356],[32422],
          [26656],[27150],[28916],
          
          [28643],[31321],[34662],
          [37498],[39484],[35561],
          [22857],[25087],[28978],
          [30504],[29461],[27542]
        ]

y_min_data = [
#         [0.10000],[0.10000],[0.10000],
#           [0.10000],[0.10000],[0.10000],
#           [0.10500],[0.12000],[0.16000],
#           [0.24400],[0.25000],
          
#           [0.10000],
          [11600],[14200],
          [18000],[20100],[21100],
          [22100],[20100],[15000],
          [15500],[17000],[18500],
          
          [18000],[19200],[23500],
          [27200],[30100],[31600],
          [32000],[30400],[20500],
          [23100],[25700],[27700],
          
          [27500],[30100],[33600],
          [36000],[34800],[25000],#xx
          [19000],[23500],[27200],
          [27000],[25500],[24000]
        ]
  
# 特征数
featurenum = 4
x = tf.placeholder(tf.float32, [None, featurenum])  
y = tf.placeholder(tf.float32, [None, 1]) 
  
#定义神经网络中间层权值  
weights_l1 = tf.Variable(tf.random_normal([featurenum, 10]))#10个神经元  
biases_l1 = tf.Variable(tf.zeros([1, 10]))  
# prediction = tf.matmul(x, weights_l1) + biases_l1  
wx_plust_b_l1 = tf.matmul(x, weights_l1) + biases_l1  
# l1 = tf.nn.relu(wx_plust_b_l1)#双曲正切函数作为激活函数  
# l1 = tf.nn.tanh(wx_plust_b_l1)#双曲正切函数作为激活函数  
# l1 = tf.sigmoid(wx_plust_b_l1)
  

#定义输出层  
weights_l2 = tf.Variable(tf.random_normal([10, 1]))#输出层1个神经元  
biases_l2 = tf.Variable(tf.zeros([1,1]))#一个偏置  
prediction = tf.matmul(wx_plust_b_l1, weights_l2) + biases_l2  
# wx_plust_b_l2 = tf.matmul(wx_plust_b_l1, weights_l2) + biases_l2  
# prediction = tf.nn.relu(wx_plust_b_l2)#预测结果  
# prediction = tf.nn.tanh(wx_plust_b_l2)#预测结果  
# prediction = tf.sigmoid(wx_plust_b_l2)#预测结果  

# tf.add_to_collection(tf.GraphKeys.WEIGHTS, biases_l1)
# # tf.add_to_collection(tf.GraphKeys.WEIGHTS, biases_l2)
# regularizer = tf.contrib.layers.l2_regularizer(scale=5.0/50000)
# reg_term = tf.contrib.layers.apply_regularization(regularizer) 
  
#代价函数  
loss = tf.reduce_mean(tf.square(y - prediction))  #+reg_term
#使用梯度下降  
train_step = tf.train.GradientDescentOptimizer(10000).minimize(loss)#学习率设置为0.1  
  
with tf.Session() as sess:  
    
    run_mode = 'ave'
#     run_mode = 'min'
    
    sess.run(tf.global_variables_initializer())#变量初始化，一定要做  
    if(run_mode == 'ave'):
        y_data = y_ave_data
        min_loss = 0.0001
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
        print('prediction:',sess.run(prediction, feed_dict={x:x_data, y:y_data}))
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
    prediction_value1 = sess.run(prediction, feed_dict={x: [[0.16*6,year_2017,0.20169,0.21244]]})
    print('预测2：',prediction_value1)
    
    #使用测试集计算模型的均方误差MSE
#     pred_y = sess.run(prediction, feed_dict={x: x_data,y:y_data})
#     pred_y = sess.run(prediction, feed_dict={x: [[0.16*6,year_2017,0.20169,0.21244]],y:[[0.27542]]})
#     mse = tf.reduce_mean(tf.square(pred_y - y_data))
#     print("MSE: %.4f" % sess.run(mse))
    
    #画图  
    seed_plt = 0.042
    x_plt = [[-seed_plt*23],[-seed_plt*22],[-seed_plt*21],[-seed_plt*20],[-seed_plt*19],[-seed_plt*18],[-seed_plt*17],[-seed_plt*16],[-seed_plt*15],[-seed_plt*14],[-seed_plt*13],[-seed_plt*12],
             [-seed_plt*11],[-seed_plt*10],[-seed_plt*9],[-seed_plt*8],[-seed_plt*7],[-seed_plt*6],[-seed_plt*5],[-seed_plt*4],[-seed_plt*3],[-seed_plt*2],[-seed_plt*1],
             [seed_plt*1],[seed_plt*2],[seed_plt*3],[seed_plt*4],[seed_plt*5],[seed_plt*6],[seed_plt*7],[seed_plt*8],[seed_plt*9],[seed_plt*10],[seed_plt*11],[seed_plt*12],#[seed_plt*13],
#              [seed_plt*14],
#              [seed_plt*15],[seed_plt*16],[seed_plt*17],[seed_plt*18],[seed_plt*19],[seed_plt*20],[seed_plt*21],[seed_plt*22],[seed_plt*23],[seed_plt*24]
             ]
    plt.figure()  
    prediction_plt1 = prediction_value.tolist()
    plt.scatter(x_plt, y_data)#画散点图  
#     plt.scatter(x_plt, prediction_plt1)#画预测的
    plt.plot(x_plt, prediction_plt1, 'r-', lw = 5)#画预测的实线，红色  
    prediction_plt = prediction_value.tolist()
    plt.show()
