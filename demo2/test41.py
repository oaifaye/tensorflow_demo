'''
Created on 2017年12月27日
两个特征的多元线性回归模型
@author: Administrator
http://bluewhale.cc/2017-08-10/use-tensorflow-to-train-linear-regression-models-and-predict.html#codesyntax_10
'''
import tensorflow as tf 
import numpy as np 
# import matplotlib.pyplot as plt

#设置广告花费，曝光与点击数据
invest=np.array([[13],[105],[105], [24], [3], [45], [35], [24], [40], [32]]).astype(np.float32)
impressions=np.array([[202],[244],[233], [175], [10], [227], [234], [216], [220], [213]]).astype(np.float32)
click=np.array([[8], [13], [11], [7],[ 2], [12], [10], [9], [11], [10]]).astype(np.float32)

#粗糙划分训练集和测试集数据
X_test = invest[0:5].reshape(-1,1)
X_test1 = impressions[0:5].reshape(-1,1)
y_test = click[0:5]
X_train = invest[5:].reshape(-1,1)
X_train1 =impressions[5:].reshape(-1,1)
y_train = click[5:]

#设置第一个自变量x的占位符
x=tf.placeholder(tf.float32,[None,1])
#设置第二个自变量x2的占位符
x2=tf.placeholder(tf.float32,[None,1])
#设置第一个斜率(权重值)W变量
W=tf.Variable(tf.zeros([1,1]))
#设置第二个斜率(权重值)W2变量
W2=tf.Variable(tf.zeros([1,1]))
#设置截距(偏置量)b变量
b=tf.Variable(tf.zeros([1]))

#设置多元线性回归模型y=Wx+W2x2+b
y=tf.matmul(x,W)+tf.matmul(x2,W2)+b
#设置占位符用于输入实际的y值
y_=tf.placeholder(tf.float32,[None,1])
#设置成本函数(最小方差)
cost=tf.reduce_mean(tf.square(y_-y))
#使用梯度下降以0.000001的学习速率最小化成本函数cost，以获得W,W2和b的值
train_step=tf.train.GradientDescentOptimizer(0.000001).minimize(cost)

#开始训练前对变量进行初始化
init=tf.global_variables_initializer()
#创建一个会话(Sess)
sess=tf.Session()
#在Sess中启用模型并初始化变量
sess.run(init)
#创建一个空list用于存放成本函数的变化
cost_history=[]
#循环训练模型1000次
for i in range(1000):
    feed={x:X_train,x2:X_train1,y_:y_train}
    sess.run(train_step,feed_dict=feed)
    #存储每次训练的cost值
    cost_history.append(sess.run(cost,feed_dict=feed))
    #输出每次训练后的W,W2,b和cost值
    print("After %d iteration:" %i)
    print("W: %f" % sess.run(W))
    print("W2 Value: %f" % sess.run(W2))
    print("b: %f" % sess.run(b))
    print("cost: %f" % sess.run(cost,feed_dict=feed))
#输出最终的W,W2,b和cost值
print("W_Value: %f" % sess.run(W),"W2 Value: %f" % sess.run(W2),"b_Value: %f" % sess.run(b),"cost_Value: %f" % sess.run(cost,feed_dict=feed))

#使用模型进行预测
print(sess.run(y, feed_dict={x: [[13]],x2:[[202]]}))

#使用测试集计算模型的均方误差MSE
pred_y = sess.run(y, feed_dict={x: X_test,x2:X_test1})
mse = tf.reduce_mean(tf.square(pred_y - y_test))
print("MSE: %.4f" % sess.run(mse))