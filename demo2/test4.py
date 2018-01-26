'''
Created on 2017年12月27日
一个特征的简单线性回归模型
@author: Administrator

http://bluewhale.cc/2017-08-10/use-tensorflow-to-train-linear-regression-models-and-predict.html#codesyntax_10
'''
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

#设置广告花费和点击量数据
money=np.array([[109],[82],[99], [72], [87], [78], [86], [84], [94], [57]]).astype(np.float32)
click=np.array([[11], [8], [8], [6],[ 7], [7], [7], [8], [9], [5]]).astype(np.float32)

#粗糙划分训练集和测试集数据
X_test = money[0:5].reshape(-1,1)
y_test = click[0:5]
X_train = money[5:].reshape(-1,1)
y_train = click[5:]

#设置自变量x的占位符,梯度下降时真实数据输入到模型的入口点
x=tf.placeholder(tf.float32,[None,1])
#设置斜率(权重值)W变量
W1=tf.Variable(tf.zeros([1,1]))
#设置截距(偏置量)b变量
b1=tf.Variable(tf.zeros([1]))
#设置线性模型y=Wx+b
y1=tf.matmul(x,W1)+b1
y11 = tf.nn.relu(y1)

'''输出层'''
#设置斜率(权重值)W变量
W2=tf.Variable(tf.random_normal([1,1]))
#设置截距(偏置量)b变量
b2=tf.Variable(tf.random_normal([1]))
#设置线性模型y=Wx+b
y=tf.matmul(y11,W2)+b2

# y = tf.nn.relu(y1)

#设置占位符用于输入实际的y值
y_=tf.placeholder(tf.float32,[None,1])

#设置成本函数(最小方差)
cost=tf.reduce_sum(tf.pow((y_-y),2))
#使用梯度下降，以0.000001的学习速率最小化成本函数cost，以获得W和b的值
train_step=tf.train.GradientDescentOptimizer(0.000001).minimize(cost)

#开始训练前对变量进行初始化
init=tf.global_variables_initializer()
#创建一个会话(Sess)
sess=tf.Session()
#在Sess中启用模型并初始化变量
sess.run(init)

#创建一个空list用于存放成本函数的变化
cost_history=[]

#循环训练模型100次
for i in range(100):
    feed={x:X_train,y_:y_train}
    sess.run(train_step,feed_dict=feed)
    #存储每次训练的cost值
    cost_history.append(sess.run(cost,feed_dict=feed))
    #输出每次训练后的W,b和cost值
    print("After %d iteration:" %i)
    print("W: %f" % sess.run(W1))
    print("b: %f" % sess.run(b1))
    print("cost: %f" % sess.run(cost,feed_dict=feed))
#输出最终的W,b和cost值
print("W_Value: %f" % sess.run(W1),"b_Value: %f" % sess.run(b1),"cost_Value: %f" % sess.run(cost,feed_dict=feed))

#绘制成本函数cost在100次训练中的变化情况
# plt.plot(range(len(cost_history)),cost_history)
# plt.axis([0,100,0,np.max(cost_history)])
# plt.xlabel('training epochs')
# plt.ylabel('cost')
# plt.title('cost history')
# plt.show()

#使用模型进行预测
print(sess.run(y, feed_dict={x: [[84]]}))

#使用测试集计算模型的均方误差MSE
pred_y = sess.run(y, feed_dict={x: X_test})
mse = tf.reduce_mean(tf.square(pred_y - y_test))
print("MSE: %.4f" % sess.run(mse))