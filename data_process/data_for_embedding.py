import os
import re
from pandas import DataFrame as df
import numpy as np

source_path_neg = 'D:\\my_work\\original_data\\test\\neg\\'
source_path_pos = 'D:\\my_work\\original_data\\test\\pos\\'

save_path = 'D:\\my_work\\transfered_data\\'
reviews_save_name = 'reviews.txt'

punc = '"#$%&\'()*+,-./:;<=>@[\\]_`{|}'


def text_cleaner(text):
    '''
    清除HTML内容
                '''
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, ' ', text)

    '''
    清除标点符号及数字
                      '''
    text = "".join([char for char in text if char not in punc])
    text = re.sub('[0-9]+', '', text)

    return text


def get_review(file):
    with open(file, 'r', encoding='utf-8') as f:
        review = f.read().strip()
        review = text_cleaner(review)  # 数据清洗

        return review


def reviews_to_txt(reviews):

    reviews = np.array(reviews)
    np.random.shuffle(reviews)

    full_name = save_path + reviews_save_name
    with open(full_name, 'a', encoding='utf-8') as f:
        counter = 0
        for review in reviews:
            f.write(review + '\n')
            counter += 1

    print(str(counter) + '条评论文本存储完成！')


if __name__ == "__main__":

    neg_file_names = os.listdir(source_path_neg)
    pos_file_names = os.listdir(source_path_pos)  # 分别获取负面、正面原始文件文件名

    '''
    获取有标注原始文件的评论内容（并做清洗与标准化）、星级、标签
    依次加载到reviews_sup stars labels 三个列表 
                                               '''

    reviews_sup = []

    print('准备加载有标注测试数据…')

    for i in range(12500):
        neg_file_name = neg_file_names[i]
        pos_file_name = pos_file_names[i]

        '''
        获取评论数据
                    '''

        neg_review = get_review(source_path_neg + neg_file_name)
        pos_review = get_review(source_path_pos + pos_file_name)

        reviews_sup.append(neg_review)
        reviews_sup.append(pos_review)



    print('准备存储测试样本到txt文件…')

    reviews_to_txt(reviews_sup)