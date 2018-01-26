'''
Created on 2017年12月26日

@author: Administrator
'''
import tensorflow as tf
import os
#初始化变量和模型参数，定义训练闭环运算
W = tf.Variable(tf.zeros([2, 1]), name="weights")#变量权值
b = tf.Variable(0., name="bias")#线性函数常量，模型偏置
def inference(X):#计算返回推断模型输出(数据X)
    print ("function: inference")
    return tf.matmul(X, W) + b
def loss(X, Y):#计算损失(训练数据X及期望输出Y)
    print ("function: loss")
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))
def inputs():#读取或生成训练数据X及期望输出Y
    print ("function: inputs")
    # Data from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)
def train(total_loss):#训练或调整模型参数(计算总损失)
    print ("function: train")
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
def evaluate(sess, X, Y):#评估训练模型
    print ("function: evaluate")
    print (sess.run(inference([[80., 25.]])))# ~ 303
    print (sess.run(inference([[65., 25.]])))# ~ 256
saver = tf.train.Saver()#创建Saver对象
#会话对象启动数据流图，搭建流程
with tf.Session() as sess:
    print ("Session: start")
    tf.global_variables_initializer().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    training_steps = 1000#实际训练迭代次数
    initial_step = 0
    checkpoin_dir = "./"
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoin_dir))
    if ckpt and ckpt.model_checkpoint_path:
        print ("checkpoint_path: " + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)#从检查点恢复模型参数
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
    for step in range(initial_step, training_steps):#实际训练闭环
        sess.run([train_op])
    if step % 10 == 0:#查看训练过程损失递减
        print (str(step)+ " loss: ", sess.run([total_loss]))
        save_file = saver.save(sess, 'my-model', global_step=step)#创建遵循命名模板my-model-{step}检查点文件
        print (str(step) + " save_file: ", save_file)
    evaluate(sess, X, Y)#模型评估
    coord.request_stop()
    coord.join(threads)
    saver.save(sess, 'my-model', global_step=training_steps)
    print (str(training_steps) + " final loss: ", sess.run([total_loss]))
    sess.close()