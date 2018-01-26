'''
Created on 2017年12月26日

@author: Administrator

http://blog.csdn.net/wulex/article/details/66972720
'''
import tensorflow as tf
import numpy as np


# def make_layer1(inputs, in_size, out_size, activate=None):
#     weights = tf.Variable(tf.random_normal([in_size, out_size]))
#     print(type(weights))
# #     basis = tf.Variable(tf.zeros([1, out_size]) + 0.1)
#     result = tf.matmul(inputs, weights)# + basis
#     if activate is None:
#         print("result：",result)
#         return result
#     else:
#         print("activate(result)")
#         return activate(result)


class BPNeuralNetwork:
    def __init__(self):
        self.session = tf.Session()
        self.input_layer = None
        self.label_layer = None
        self.loss = None
        self.optimizer = None
        self.layers = []
        self.weights = None
        self.basis = None
        self.result = None
        self.weights1 = None
        self.basis1 = None
        self.result1 = None

    def __del__(self):
        self.session.close()
        
    def make_layer(self,inputs, in_size, out_size, activate=None):
        self.weights = tf.Variable(tf.random_normal([in_size, out_size]))
        self.basis = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        self.result = tf.matmul(inputs, self.weights) + self.basis
        if activate is None:
            return self.result
        else:
            return activate(self.result)
        
    def make_layer1(self,inputs, in_size, out_size, activate=None):
        self.weights1 = tf.Variable(tf.random_normal([in_size, out_size]))
        self.basis1 = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        self.result1 = tf.matmul(inputs, self.weights1) + self.basis1
        if activate is None:
            return self.result1
        else:
            return activate(self.result1)

    def train(self, cases, labels, limit=100, learn_rate=0.05):
        # 构建网络
        self.input_layer = tf.placeholder(tf.float32, [None, 2])
        self.label_layer = tf.placeholder(tf.float32, [None, 1])
        self.layers.append(self.make_layer(self.input_layer, 2, 10, activate=tf.nn.relu))
        self.layers.append(self.make_layer1(self.layers[0], 10, 2, activate=None))
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square((self.label_layer - self.layers[1])), reduction_indices=[1]))
        self.optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        initer = tf.initialize_all_variables()
        # 做训练
        self.session.run(initer)
        for i in range(limit):
            print('self.input_layer:',self.session.run(self.input_layer, feed_dict={self.input_layer: cases, self.label_layer: labels}))
            print()
            print('self.label_layer:',self.session.run(self.label_layer, feed_dict={self.input_layer: cases, self.label_layer: labels}))
            print()
            print('self.weights:',self.session.run(self.weights))
            print()
            print('self.basis:',self.session.run(self.basis))
            print()
            print('self.result:',self.session.run(self.result,feed_dict={self.input_layer: cases, self.label_layer: labels}))
            print()
            print('self.layers[0]:',self.session.run(self.layers[0], feed_dict={self.input_layer: cases, self.label_layer: labels}))
            print()
            print('self.layers[1]:',self.session.run(self.layers[1], feed_dict={self.input_layer: cases, self.label_layer: labels}))
            print()
            print('self.loss:',self.session.run(self.loss, feed_dict={self.input_layer: cases, self.label_layer: labels}))
#             print(self.session.run(self.layers[0], feed_dict={self.input_layer: cases, self.label_layer: labels}))
            self.session.run(self.optimizer, feed_dict={self.input_layer: cases, self.label_layer: labels})

    def predict(self, case):
        return self.session.run(self.layers[1], feed_dict={self.input_layer: case})

    def test(self):
        x_data = np.array([[0, 0], [0, 2], [2, 0], [2, 2]])
        y_data = np.array([[0, 2, 2, 0]]).transpose()
        test_data = np.array([[2,0]])
        self.train(x_data, y_data)
        print(self.predict(test_data))

nn = BPNeuralNetwork()
nn.test()