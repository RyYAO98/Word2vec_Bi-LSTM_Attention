#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：一个类，执行文本转换
输入：分词文本
输出：句子列表，全文的词汇列表，TF，DF
时间：2016年5月17日 19:08:34
"""

import codecs
import re
from tkinter.filedialog import askopenfilename


class TextSta:
    # 定义基本属性，分词文本的全路径
    filename = ""

    # 定义构造方法
    def __init__(self, path):    # 参数path，赋给filename
        self.filename = path

    def sen(self):    # 获取句子列表
        with open(self.filename, "r", encoding="utf-8") as f1:
            print(u"已经打开文本：", self.filename)

        # 获得句子列表，其中每个句子又是词汇的列表
            sentences_list = []
            for line in f1.readlines():
                single_sen_list = line.strip().split(" ")
                while "" in single_sen_list:
                    single_sen_list.remove("")
                sentences_list.append(single_sen_list)
            print(u"句子总数：", len(sentences_list))

        return sentences_list

if __name__ == "__main__": 
    pass
