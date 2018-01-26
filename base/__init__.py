import tensorflow
import numpy as np
from pip._vendor.requests.packages.urllib3.connectionpool import xrange

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
# x_data = np.float32(np.random.rand(2, 3)) # 随机输入
# x_data = [[1,2,3],[4,5,6]]
# y_data = np.dot([10, 20], x_data) 
# print(111)

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
# 
b = tensorflow.Variable(tensorflow.zeros([1]))
W = tensorflow.Variable(tensorflow.random_uniform([1, 2], -1.0, 1.0))
y = tensorflow.matmul(W, x_data) + b

# 最小化方差
loss = tensorflow.reduce_mean(tensorflow.square(y - y_data))
optimizer = tensorflow.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tensorflow.initialize_all_variables()

# 启动图 (graph)
sess = tensorflow.Session()
sess.run(init)

# 拟合平面
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]