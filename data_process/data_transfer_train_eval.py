import os
import re
import nltk
from pandas import DataFrame as df
import numpy as np

source_path_neg = 'D:\\my_work\\original_data\\train\\neg\\'
source_path_pos = 'D:\\my_work\\original_data\\train\\pos\\'
source_path_unsup = 'D:\\my_work\\original_data\\train\\unsup\\'

save_path = 'D:\\my_work\\transfered_data\\'
samples_save_prefix = 'sample_' # 用于存储训练样本csv文件
reviews_save_name = 'reviews.txt' # 用于存储所有经过处理的评论文本的txt文件

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
    text  = "".join([char for char in text if char not in punc])
    text = re.sub('[0-9]+', '', text)

    return text


def get_review(file):
    
    with open(file, 'r', encoding='utf-8') as f:
        review = f.read().strip()
        review = text_cleaner(review) # 数据清洗
        
        return review

def data_to_csv(c1, c2, c3):

    data_dict = {'reviews': c1,
                 'stars': c2,
                 'labels': c3}
    data_frame = df(data_dict)
    data_frame_shuffled = data_frame.sample(frac=1).reset_index(drop=True) # 对数据进行按行打乱

    sample_size = 5000 # 定义交叉验证的单位样本集容量

    for i in range(0, 5):

        sample_data = data_frame_shuffled.ix[i*sample_size:(i+1)*sample_size-1]

        full_name = save_path + samples_save_prefix + str(i+1) + '.csv'
        sample_data.to_csv(full_name, index=0)

    print('训练样本数据存储完成！')

def reviews_to_txt(reviews):

    reviews = np.array(reviews)
    np.random.shuffle(reviews)
    
    full_name = save_path + reviews_save_name
    with open(full_name, 'a', encoding='utf-8') as f:
        counter = 0
        for review in reviews:
            f.write(review + '\n')
            counter += 1

    print(str(counter)+'条评论文本存储完成！')    
        

if __name__ == "__main__":

    neg_file_names = os.listdir(source_path_neg)
    pos_file_names = os.listdir(source_path_pos)
    unsup_file_names = os.listdir(source_path_unsup) # 分别获取负面、正面、未标注原始文件文件名

    '''
    获取有标注原始文件的评论内容（并做清洗与标准化）、星级、标签
    依次加载到reviews_sup stars labels 三个列表 
                                               '''
    
    reviews_sup = []
    stars = []
    labels = [0, 1] * 12500 # 每次迭代依次加载一条差评及一条好评

    print('准备加载有标注数据…')
    
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
        
        '''
        获取星级数据
                    '''

        neg_star = int(neg_file_name.split('.')[0].split('_')[1])
        pos_star = int(pos_file_name.split('.')[0].split('_')[1])

        stars.append(neg_star)
        stars.append(pos_star)

    '''
    获取无标注原始文件的评论内容，做清洗与标准化
    加载处理后的内容到reviews_unsup
                                   '''
    
    reviews_unsup = []

    print('准备加载无标注数据…')

    for i in range(50000):

        unsup_file_name = unsup_file_names[i]
        unsup_review = get_review(source_path_unsup + unsup_file_name)
        reviews_unsup.append(unsup_review)

    reviews = reviews_sup + reviews_unsup

    print('准备存储训练样本到csv文件…')
    data_to_csv(reviews_sup, stars, labels) # 存储训练样本到csv文件
    print('准备存储评论文本到txt文件…')
    reviews_to_txt(reviews) # 存储所有经过处理的评论文本到txt文件

        
    
