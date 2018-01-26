# -*- coding:utf-8 -*- 

'''
    词向量：使用tf来实现word2vec
    http://blog.csdn.net/u014595019/article/details/54093161
    git:https://github.com/multiangle/tfword2vec
'''

import tensorflow as tf
import numpy as np
import math
import collections
import pickle as pkl
from pprint import pprint
import re
import jieba
import os.path as path
import os
from tensorflow.python.keras._impl.keras.models import save_model

class word2vec():
    def __init__(self,
                 vocab_list=None,
                 embedding_size=200,
                 win_len=3, # 单边窗口长
                 num_sampled=1000,
                 learning_rate=1.0,
                 logdir='/tmp/simple_word2vec',
                 model_path= None
                 ):

        # 获得模型的基本参数
        self.batch_size     = None # 一批中数据个数, 目前是根据情况来的
        if model_path!=None:
            self.load_model(model_path)
        else:
            # model parameters
            assert type(vocab_list)==list
            self.vocab_list     = vocab_list
            self.vocab_size     = vocab_list.__len__()
            self.embedding_size = embedding_size
            self.win_len        = win_len
            self.num_sampled    = num_sampled
            self.learning_rate  = learning_rate
            self.logdir         = logdir

            self.word2id = {}   # word => id 的映射
            for i in range(self.vocab_size):
                self.word2id[self.vocab_list[i]] = i

            # train times
            self.train_words_num = 0 # 训练的单词对数
            self.train_sents_num = 0 # 训练的句子数
            self.train_times_num = 0 # 训练的次数（一次可以有多个句子）

            # train loss records
            self.train_loss_records = collections.deque(maxlen=10) # 保存最近10次的误差
            self.train_loss_k10 = 0

        self.build_graph()
        self.init_op()
        if model_path!=None:
            tf_model_path = os.path.join(model_path,'tf_vars')
            self.saver.restore(self.sess,tf_model_path)

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)


    '''
            核心代码
    '''
    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 首先定义两个用作输入的占位符，分别输入输入集(train_inputs)和标签集(train_labels)
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            
            # 词向量矩阵，初始时为均匀随机正态分布
            self.embedding_dict = tf.Variable(
                tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0)
            )
            
            # 模型内部参数矩阵，初始为截断正太分布
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                              stddev=1.0/math.sqrt(self.embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # 将输入序列向量化
            embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs) # batch_size

            # 得到NCE损失(负采样得到的损失)
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights = self.nce_weight,
                    biases = self.nce_biases,
                    labels = self.train_labels,
                    inputs = embed,
                    num_sampled = self.num_sampled,
                    num_classes = self.vocab_size
                )
            )

            # tensorboard 相关
            tf.summary.scalar('loss',self.loss)  # 让tensorflow记录参数

            # 根据 nce loss 来更新梯度和embedding，使用梯度下降法(gradient descent)来实现
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)  # 训练操作

            # 计算与指定若干单词的相似度
            self.test_word_id = tf.placeholder(tf.int32,shape=[None])
            vec_l2_model = tf.sqrt(  # 求各词向量的L2模
                tf.reduce_sum(tf.square(self.embedding_dict),1,keep_dims=True)
            )

            avg_l2_model = tf.reduce_mean(vec_l2_model)
            tf.summary.scalar('avg_vec_model',avg_l2_model)

            self.normed_embedding = self.embedding_dict / vec_l2_model
            # self.embedding_dict = norm_vec # 对embedding向量正则化
            test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)
            self.similarity = tf.matmul(test_embed, self.normed_embedding, transpose_b=True)

            # 变量初始化
            self.init = tf.global_variables_initializer()
            # 汇总所有的变量记录
            self.merged_summary_op = tf.summary.merge_all()
            # 保存模型的操作
            self.saver = tf.train.Saver()

    def train_by_sentence(self, input_sentence=[]):
        #  input_sentence: [sub_sent1, sub_sent2, ...]
        # 每个sub_sent是一个单词序列，例如['这次','大选','让']
        sent_num = input_sentence.__len__()
        batch_inputs = []
        batch_labels = []
        for sent in input_sentence:# 输入有可能是多个句子，这里每个循环处理一个句子
            for i in range(sent.__len__()): # 处理单个句子中的每个单词
                start = max(0,i-self.win_len)# 窗口为 [-win_len,+win_len],总计长2*win_len+1
                end = min(sent.__len__(),i+self.win_len+1)
                # 将某个单词对应窗口中的其他单词转化为id计入label，该单词本身计入input
                for index in range(start,end):
                    if index == i:
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])
                        label_id = self.word2id.get(sent[index])
                        if not (input_id and label_id):#　如果单词不在词典中，则跳过
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        if len(batch_inputs)==0:#　如果标签集为空，则跳过
            return
        batch_inputs = np.array(batch_inputs,dtype=np.int32)
        batch_labels = np.array(batch_labels,dtype=np.int32)
        batch_labels = np.reshape(batch_labels,[batch_labels.__len__(),1])

        #　生成供tensorflow训练用的数据
        feed_dict = {
            self.train_inputs: batch_inputs,
            self.train_labels: batch_labels
        }
        
        # 这句操控tf进行各项操作。数组中的选项，train_op等，是让tf运行的操作，feed_dict选项用来输入数据
        _, loss_val, summary_str = self.sess.run([self.train_op,self.loss,self.merged_summary_op], feed_dict=feed_dict)

        # train loss，记录这次训练的loss值
        self.train_loss_records.append(loss_val)
        # self.train_loss_k10 = sum(self.train_loss_records)/self.train_loss_records.__len__()
        self.train_loss_k10 = np.mean(self.train_loss_records)# 求loss均值
        if self.train_sents_num % 1000 == 0 :
            self.summary_writer.add_summary(summary_str,self.train_sents_num)
            print("{a} sentences dealed, loss: {b}"
                  .format(a=self.train_sents_num,b=self.train_loss_k10))

        # train times
        self.train_words_num += batch_inputs.__len__()
        self.train_sents_num += input_sentence.__len__()
        self.train_times_num += 1

    def cal_similarity(self,test_word_id_list,top_k=10):
        sim_matrix = self.sess.run(self.similarity, feed_dict={self.test_word_id:test_word_id_list})
        sim_mean = np.mean(sim_matrix)
        sim_var = np.mean(np.square(sim_matrix-sim_mean))
        test_words = []
        near_words = []
        for i in range(test_word_id_list.__len__()):
            test_words.append(self.vocab_list[test_word_id_list[i]])
            nearst_id = (-sim_matrix[i,:]).argsort()[1:top_k+1]
            nearst_word = [self.vocab_list[x] for x in nearst_id]
            near_words.append(nearst_word)
        return test_words,near_words,sim_mean,sim_var

    def save_model(self, save_path):

        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # 记录模型各参数
        model = {}
        var_names = ['vocab_size',      # int       model parameters
                     'vocab_list',      # list
                     'learning_rate',   # int
                     'word2id',         # dict
                     'embedding_size',  # int
                     'logdir',          # str
                     'win_len',         # int
                     'num_sampled',     # int
                     'train_words_num', # int       train info
                     'train_sents_num', # int
                     'train_times_num', # int
                     'train_loss_records',  # int   train loss
                     'train_loss_k10',  # int
                     ]
        for var in var_names:
            model[var] = eval('self.'+var)

        param_path = os.path.join(save_path,'params.pkl')
        if os.path.exists(param_path):
            os.remove(param_path)
        with open(param_path,'wb') as f:
            pkl.dump(model,f)

        # 记录tf模型
        tf_path = os.path.join(save_path,'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess,tf_path)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise RuntimeError('file not exists')
        param_path = os.path.join(model_path,'params.pkl')
        with open(param_path,'rb') as f:
            model = pkl.load(f)
            self.vocab_list = model['vocab_list']
            self.vocab_size = model['vocab_size']
            self.logdir = model['logdir']
            self.word2id = model['word2id']
            self.embedding_size = model['embedding_size']
            self.learning_rate = model['learning_rate']
            self.win_len = model['win_len']
            self.num_sampled = model['num_sampled']
            self.train_words_num = model['train_words_num']
            self.train_sents_num = model['train_sents_num']
            self.train_times_num = model['train_times_num']
            self.train_loss_records = model['train_loss_records']
            self.train_loss_k10 = model['train_loss_k10']


def doTrain(txt_path,model_path):
    # step 1 读取停用词
    stop_words = []
    with open('stop_words.txt',encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))

    # step2 读取文本，预处理，分词，得到词典
    raw_word_list = []#全部词汇
    sentence_list = []#已行为单位的词汇组成的list
    with open(txt_path,encoding='utf-8') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n','')
            while ' ' in line:
                line = line.replace(' ','')
            if len(line)>0: # 如果句子非空
                raw_words = list(jieba.cut(line,cut_all=False))
                dealed_words = []
                for word in raw_words:
                    if word not in stop_words and word not in ['qingkan520','www','com','http']:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line = f.readline()
    word_count = collections.Counter(raw_word_list)
    print('文本中总共有{n1}个单词,不重复单词数{n2},选取前50000个单词进入词典'
          .format(n1=len(raw_word_list),n2=len(word_count)))
    word_count = word_count.most_common(50000)
    word_list = [x[0] for x in word_count]

    # 创建模型，训练
    w2v = word2vec(vocab_list=word_list,    # 词典集
                   embedding_size=200,
                   win_len=2,
                   learning_rate=1,
                   num_sampled=100,         # 负采样个数
                   logdir='D:/tmp/280',
#                    model_path='D:/tmp/mod'
                   )       # tensorboard记录地址
    num_steps = 100000
    for i in range(num_steps):
        mo = i%len(sentence_list)
        sent = sentence_list[mo]
        w2v.train_by_sentence([sent])
        
    w2v.save_model(model_path)
        
    #测试
#     test_word = ['萧炎','灵魂','火焰','长老','尊者','皱眉']
#     test_id = [word_list.index(x) for x in test_word]
#     test_words,near_words,sim_mean,sim_var = w2v.cal_similarity(test_id,5)
#     for test_word in test_words:
#         print('test_word:',test_word)
#     for near_word in near_words:
#         print('near_word:',near_word)
        
def doTest(txt_path,model_path,test_word):
    # step 1 读取停用词
    stop_words = []
    with open('stop_words.txt',encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))

    # step2 读取文本，预处理，分词，得到词典
    raw_word_list = []
    sentence_list = []
    with open(txt_path,encoding='utf-8') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n','')
            while ' ' in line:
                line = line.replace(' ','')
            if len(line)>0: # 如果句子非空
                raw_words = list(jieba.cut(line,cut_all=False))
                dealed_words = []
                for word in raw_words:
                    if word not in stop_words and word not in ['qingkan520','www','com','http']:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line = f.readline()
    word_count = collections.Counter(raw_word_list)
    print('文本中总共有{n1}个单词,不重复单词数{n2},选取前50000个单词进入词典'
          .format(n1=len(raw_word_list),n2=len(word_count)))
    #取一个list里前多少个
    word_count = word_count.most_common(50000)
    word_list = [x[0] for x in word_count]
    
    # 创建模型，测试
    w2v = word2vec(vocab_list=word_list,    # 词典集
                   embedding_size=200,
                   win_len=2,
                   learning_rate=1,
                   num_sampled=100,         # 负采样个数
                   logdir='D:/tmp/280',
#                    model_path='D:/tmp/mod'
                   )    
    w2v.load_model(model_path)   
    
    #测试
#     test_word = ['萧炎','灵魂','火焰','长老','尊者','皱眉']
    test_id = [word_list.index(x) for x in test_word]
    test_words,near_words,sim_mean,sim_var = w2v.cal_similarity(test_id,10)
    for test_word in test_words:
        print('test_word:',test_word)
    for near_word in near_words:
        print('near_word:',near_word)

if __name__=='__main__':
#     model_path = 'D:/tmp/mod'
#     txt_path = '280.txt'
#     doTest(txt_path,model_path)
    
#     model_path = 'D:/tmp/san'
#     txt_path = 'san.txt'
#     test_word = ['司马懿','曹操','刘备','诸葛亮','关羽']
# #     doTrain(txt_path,model_path)
#     doTest(txt_path,model_path,test_word)
    
#     model_path = 'D:/tmp/gui'
#     txt_path = 'gui.txt'
#     test_word = ['胡八一','盗墓','粽子']
# #     doTrain(txt_path,model_path)
#     doTest(txt_path,model_path,test_word)
    
    model_path = 'D:/tmp/jpm'
    txt_path = 'jpm.txt'
    test_word = ['潘金莲','金莲','西门庆','西门庆']
    doTrain(txt_path,model_path)
    doTest(txt_path,model_path,test_word)
        
        
        
    
#     w2v.save_model("/tmp/res/")
#进入log文件的上级目录（本例中即D:/tmp目录），在路径栏中直接输入cmd启动dos对话框。
#tensorboard --logdir=280