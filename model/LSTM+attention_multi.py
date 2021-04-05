# coding=utf-8
import os
import csv
import time
import datetime
import random
import json
import math
import warnings
from collections import Counter
from math import sqrt

import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from pandas import DataFrame

warnings.filterwarnings("ignore")


# 配置参数
def starLoosen(star):
    status = {
              1: [1, 2],
              2: [3, 4],
              3: [7, 8],
              4: [9, 10]
    }
    for i in status.keys():
        if star in status[i]:
            star = i

    return star

class TrainingConfig(object):
    epoches = 5
    evaluateEvery = 157
    checkpointEvery = 157
    learningRate = 0.0005


class ModelConfig(object):
    embeddingSize = 200

    hiddenSizes = [128]  # LSTM结构的神经元个数

    dropoutKeepProb = 0.5
    l2RegLambda = 0


class Config(object):
    sequenceLength = 230  # 取了所有序列长度的均值
    batchSize = 128

    dataSource = ["D:\\my_work\\transfered_data\\sample_1.csv",
                  "D:\\my_work\\transfered_data\\sample_2.csv",
                  "D:\\my_work\\transfered_data\\sample_3.csv",
                  "D:\\my_work\\transfered_data\\sample_4.csv",
                  "D:\\my_work\\transfered_data\\sample_5.csv"]

    numClasses = 2

    training = TrainingConfig()

    model = ModelConfig()


# 实例化配置参数对象
config = Config()


# 数据预处理的类，生成训练集和测试集

class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource
        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None

        self._wordToIndex = {}
        self._indexToWord = {}

    def _readData(self, dataSource, cross_time):
        """
        从csv文件中读取数据集
        """
        evalDataSource = dataSource[cross_time]
        dataSource.remove(dataSource[cross_time])
        trainDataSource = dataSource

        trainReview = []
        trainLabels = []

        for i in range(len(trainDataSource)):
            df = pd.read_csv(trainDataSource[i])
            trainReview = trainReview + df['reviews'].tolist()
            for j in range(5000):
                trainLabels.append([starLoosen(df.ix[j, 'stars'])])  # 标签也须存储在二维列表中

        trainReviews = [line.strip().split() for line in trainReview]

        df = pd.read_csv(evalDataSource)
        evalReview = df['reviews'].tolist()
        evalReviews = [line.strip().split() for line in evalReview]
        evalLabels = []
        for j in range(5000):
            evalLabels.append([starLoosen(df.ix[j, 'stars'])])

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _reviewProcess(self, review, sequenceLength, wordToIndex):
        """
        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        """

        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength

        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)

        for i in range(sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex["UNK"]

        return reviewVec

    def _genTrainEvalData(self, x1, x2, y1, y2):
        """
        生成训练集和验证集
        """
        trainReviews = []
        evalReviews = []

        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x1)):
            reviewVec = self._reviewProcess(x1[i], self._sequenceLength, self._wordToIndex)
            trainReviews.append(reviewVec)

        for i in range(len(x2)):
            reviewVec = self._reviewProcess(x2[i], self._sequenceLength, self._wordToIndex)
            evalReviews.append(reviewVec)

        trainReviews = np.asarray(trainReviews, dtype="int64")
        trainLabels = np.array(y1, dtype="float32")

        evalReviews = np.asarray(evalReviews, dtype="int64")
        evalLabels = np.array(y2, dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def genVocabulary(self, reviews):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        allWords = [word for review in reviews for word in review]

        wordCount = Counter(allWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= 5]

        vocab, wordEmbedding = self._getWordEmbedding(words)  # vocab wordEmbedding index 一一对应
        self.wordEmbedding = wordEmbedding

        self._wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToWord = dict(zip(list(range(len(vocab))), vocab))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        '''(use once before store)
        with open("D:\\my_work\\wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._wordToIndex, f)

        with open("D:\\my_work\\indexToWord.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToWord, f)
                                            '''

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """

        wordVec = gensim.models.KeyedVectors.load_word2vec_format("D:\\my_work\\word2Vec.bin", binary=True)
        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("pad")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")

        return vocab, np.array(wordEmbedding)

    def dataGen(self, cross_time):
        """
        初始化训练集和验证集
        """

        # 初始化数据集
        trainReviews, trainLabels, evalReviews, evalLabels = self._readData(self._dataSource[:], cross_time)


        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(trainReviews, evalReviews,
                                                                                    trainLabels, evalLabels)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels


# 输出batch数据集

def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    numBatches = math.ceil(len(x) / batchSize)

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY


# 构建模型
class BiLSTMAttention(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, wordEmbedding):
        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None, 1], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                   self.embeddedWords, dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embeddedWords = tf.concat(outputs_, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self.embeddedWords, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self.attention(H)
            outputSize = config.model.hiddenSizes[-1]

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, 4],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[4]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")

            self.multiPreds = tf.cast((tf.argmax(self.predictions, 1) + 1), tf.float32, name="multiPreds")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predictions,
                                                                    labels=tf.reshape(self.inputY-1, [-1]))
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = config.model.hiddenSizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, config.sequenceLength])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, config.sequenceLength, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output


# 定义性能指标函数

def mean(item):
    return sum(item) / len(item)


def genMetrics(trueY, multiPredY):
    """
    生成acc
    """
    accuracy = accuracy_score(trueY, multiPredY)
    '''
    correctCounter = 0
    for i in range(config.batchSize):
        if math.fabs(trueY[i] - multiPredY[i]) < 2:
            correctCounter += 1
    accuracy = correctCounter / config.batchSize
                                                '''
    return round(accuracy, 4)


data = Dataset(config)

all_reviews_path = 'D:\\my_work\\transfered_data\\reviews.txt'
with open(all_reviews_path, 'r', encoding='utf-8') as f:
    review_list = f.readlines()
    reviews = [review.strip().split() for review in review_list]

data.genVocabulary(reviews)

# 创建交叉验证循环
for j in range(0, 5):
    print('@ Notice: Ready to start cross_time ' + str(j))
    cross_time = j
    data.dataGen(cross_time)
    # 训练模型

    # 生成训练集和验证集
    trainReviews = data.trainReviews
    trainLabels = data.trainLabels
    evalReviews = data.evalReviews
    evalLabels = data.evalLabels

    wordEmbedding = data.wordEmbedding

    # 定义计算图
    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

        sess = tf.Session(config=session_conf)

        # 定义会话
        with sess.as_default():
            lstm = BiLSTMAttention(config, wordEmbedding)

            globalStep = tf.Variable(0, name="globalStep", trainable=False)
            # 定义优化函数，传入学习速率参数
            optimizer = tf.train.AdamOptimizer(config.training.learningRate)
            # 计算梯度,得到梯度和变量
            gradsAndVars = optimizer.compute_gradients(lstm.loss)
            # 将梯度应用到变量下，生成训练器
            trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

            # 用summary绘制tensorBoard
            gradSummaries = []
            for g, v in gradsAndVars:
                if g is not None:
                    tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

            outDir = os.path.abspath(os.path.join("D:\\my_work\\", "summarys_multi"))
            print("Writing to {}\n".format(outDir))

            lossSummary = tf.summary.scalar("loss", lstm.loss)
            summaryOp = tf.summary.merge_all()

            trainSummaryDir = os.path.join(outDir, "train" + str(cross_time))
            trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

            evalSummaryDir = os.path.join(outDir, "eval" + str(cross_time))
            evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

            # 初始化所有变量
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

            sess.run(tf.global_variables_initializer())


            def trainStep(batchX, batchY):
                """
                训练函数
                """
                feed_dict = {
                    lstm.inputX: batchX,
                    lstm.inputY: batchY,
                    lstm.dropoutKeepProb: config.model.dropoutKeepProb
                }
                _, summary, step, loss, multiPreds, prediction = sess.run(
                    [trainOp, summaryOp, globalStep, lstm.loss, lstm.multiPreds, lstm.predictions],
                    feed_dict)

                timeStr = datetime.datetime.now().isoformat()
                acc = genMetrics(batchY, multiPreds)
                print("{}, step: {}, loss: {}, acc: {}".format(timeStr, step, loss, acc))
                trainSummaryWriter.add_summary(summary, step)


            def devStep(batchX, batchY):
                """
                验证函数
                """
                feed_dict = {
                    lstm.inputX: batchX,
                    lstm.inputY: batchY,
                    lstm.dropoutKeepProb: 1.0
                }
                summary, step, loss, multiPreds = sess.run(
                    [summaryOp, globalStep, lstm.loss, lstm.multiPreds],
                    feed_dict)

                acc = genMetrics(batchY, multiPreds)

                evalSummaryWriter.add_summary(summary, step)

                return loss, acc


            for i in range(config.training.epoches):
                # 训练模型
                print("start training model")
                for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                    trainStep(batchTrain[0], batchTrain[1])

                    currentStep = tf.train.global_step(sess, globalStep)
                    if currentStep % config.training.evaluateEvery == 0:
                        print("\nEvaluation:")

                        losses = []
                        accs = []

                        for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                            loss, acc = devStep(batchEval[0], batchEval[1])
                            losses.append(loss)
                            accs.append(acc)

                        time_str = datetime.datetime.now().isoformat()
                        print("{}, step: {}, loss: {}, acc: {}".format(time_str, currentStep, mean(losses), mean(accs)))

                    if currentStep % config.training.checkpointEvery == 0:
                        # 保存模型的另一种方法，保存checkpoint文件
                        path = saver.save(sess, "D:\\my_work\\model_ckpt_multi\\LSTM_Attention_" + str(cross_time) + str(currentStep) + ".ckpt")
                        print("Saved model checkpoint to {}\n".format(path))

