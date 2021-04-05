import tensorflow as tf
import pandas as pd
import numpy as np
import gensim
import warnings
import json
import math
import datetime
from collections import Counter
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")
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

class Config(object):
    sequenceLength = 230  # 取了所有序列长度的均值
    batchSize = 64
    embeddingSize = 200

    dataSource = 'E:\\my_work\\transfered_data\\sample_test.csv'

# 实例化配置参数对象
config = Config()
# 数据预处理的类，生成训练集和测试集

class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._batchSize = config.batchSize
        self._embeddingSize = config.embeddingSize

        self.testReviews = []
        self.testLabels = []

        self.wordEmbedding = None

        self._wordToIndex = {}
        self._indexToWord = {}

    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """

        df = pd.read_csv(filePath)
        label = df["stars"].tolist()
        labels = [starLoosen(star) for star in label]
        review = df["reviews"].tolist()
        reviews = [line.strip().split() for line in review]

        return reviews, labels

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

    def _genTestData(self, x, y):
        """
        生成训练集和验证集
        """

        reviews = []
        labels = []

        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            reviewVec = self._reviewProcess(x[i], self._sequenceLength, self._wordToIndex)
            reviews.append(reviewVec)

            labels.append([y[i]])

        testReviews = np.asarray(reviews, dtype="int64")
        testLabels = np.array(labels, dtype="float32")

        return testReviews, testLabels

    def _genVocabulary(self):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """
        with open('E:\\my_work\\wordToIndex.json', 'r', encoding='utf-8') as f:
            content = f.read()
            self._wordToIndex = json.loads(content)


    def dataGen(self):
        """
        初始化测试集
        """


        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)

        # 初始化词汇-索引映射表
        self._genVocabulary()

        # 初始化训练集和测试集
        testReviews, testLabels = self._genTestData(reviews, labels)
        self.testReviews = testReviews
        self.testLabels = testLabels


data = Dataset(config)
data.dataGen()

def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    numBatches = math.ceil(len(x)/batchSize)

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY

def mean(item):
    return sum(item) / len(item)

def genMetrics(trueY, multiPredY):
    """
    生成acc
    """
    accuracy = accuracy_score(trueY, multiPredY)

    return round(accuracy, 4)

testReviews = data.testReviews
testLabels = data.testLabels

saver = tf.train.import_meta_graph('E:\\my_work\\pretrained_ckpt\\sota_multi.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, 'E:\\my_work\\pretrained_ckpt\\sota_multi.ckpt')
    graph = tf.get_default_graph()

    inputX = graph.get_tensor_by_name("inputX:0")
    inputY = graph.get_tensor_by_name("inputY:0")
    dropoutKeepProb = graph.get_tensor_by_name("dropoutKeepProb:0")
    predictions = graph.get_tensor_by_name("output/predictions:0")
    multiPreds = graph.get_tensor_by_name("output/multiPreds:0")

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=tf.reshape(inputY-1, [-1]))
    loss = tf.reduce_mean(cross_entropy)

    def devStep(batchX, batchY):
        """
        验证函数
        """
        feed_dict = {
            inputX: batchX,
            inputY: batchY,
            dropoutKeepProb: 1.0
        }
        loss_var, predictions_var, multiPreds_var = sess.run(
            [loss, predictions, multiPreds],
            feed_dict)
        acc = genMetrics(batchY, multiPreds_var)

        return loss_var, acc

    losses = []
    accs = []

    for batchTest in nextBatch(testReviews, testLabels, config.batchSize):
        loss_var, acc = devStep(batchTest[0], batchTest[1])
        losses.append(loss_var)
        accs.append(acc)

    time_str = datetime.datetime.now().isoformat()

    print("{}, loss: {}, acc: {}".format(time_str, mean(losses), mean(accs)))
