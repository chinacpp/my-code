import os
import jieba
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import zhconv
import jieba
import jieba.posseg as psg
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import pickle


# 修改工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设置 jieba 不输出日志
jieba.setLogLevel(jieba.logging.INFO)


# 数据格式转换
def load_email_data():

    # 1. 将分布在不同文件中的邮件数据存储到同一个 csv 文件中
    # 2. 划分成训练集和测试集

    # 读取邮件标签、邮件路径
    filenames, labels = [], []
    with open('trec06c/full/index') as file:
        for line in file:
            label, filename = line.strip().split()
            filenames.append(filename)
            labels.append(label)


    # 根据邮件路径读取文件内容
    os.chdir('trec06c/full')
    contents = []
    for filename in filenames:
        with open(filename, encoding='gbk', errors='ignore') as file:
            content = file.read()
            contents.append(content)

    # 数据集分割
    x_train, x_test, y_train, y_test =\
        train_test_split(contents, labels, test_size=0.2, stratify=labels, random_state=42)

    # 存储到 csv 文件中
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    train_data = pd.DataFrame()
    train_data['emails'] = x_train
    train_data['labels'] = y_train
    train_data.to_csv('data/01-原始邮件数据-训练集.csv')

    test_data = pd.DataFrame()
    test_data['emails'] = x_test
    test_data['labels'] = y_test
    test_data.to_csv('data/01-原始邮件数据-测试集.csv')


# 数据清洗工作
def clean_data(email):

    # 1. 去除非中文的字符
    # 将满足某个模式的内容进行替换
    email = re.sub(r'[^\u4e00-\u9fa5]', '', email)

    # 2. 繁体转简体
    # 安装: pip install zhconv
    email = zhconv.convert(email, 'zh-cn')

    # 3. 邮件分词(词性筛选)
    email_pos = psg.cut(email)
    allow_pos = ['n', 'nr', 'ns', 'nt', 'v', 'a']
    email = []
    for word, pos in email_pos:
        if pos in allow_pos:
            email.append(word)

    # 转换成字符串
    email = ' '.join(email)

    return email


def clean_email_data():

    # 读取数据
    train_data = pd.read_csv('data/01-原始邮件数据-训练集.csv')

    # 数据清理
    emails = []
    labels = []
    progress = tqdm(range(len(train_data)), desc='清洗进度')
    for email, label in zip(train_data['emails'], train_data['labels']):

        # 数据清洗
        email = clean_data(email)
        # 将邮件内容长度为0的内容去除
        if len(email) == 0:
            continue

        # 存储清洗后的内容
        emails.append(email)
        labels.append(label)

        # 更新进度
        progress.update()


    # 数据存储
    train_data = pd.DataFrame()
    train_data['emails'] = emails
    train_data['labels'] = labels
    train_data.to_csv('data/02-清洗后的数据-训练集.csv')


# 文本特征提取
def extract_email_feature():

    # 1. 读取训练集数据
    train_data = pd.read_csv('data/02-清洗后的数据-训练集.csv')

    # 2. 提取特征+将训练集转换成词频向量
    transfer = CountVectorizer(max_features=10000)
    # 2.1 提取语料特征
    # 2.2 将训练数据转换成词频向量
    emails = transfer.fit_transform(train_data['emails'])

    # 3. 存储转换后的内容
    train_data_dict = {}
    train_data_dict['emails'] = emails.toarray().tolist()
    train_data_dict['labels'] = train_data['labels'].tolist()
    # pickle 可以将 python 对象序列化到磁盘中
    pickle.dump(train_data_dict, open('data/03-模型训练数据.pkl', 'wb'), 3)

    # 3. 存储特征
    # print(len(transfer.get_feature_names_out()))
    # 注意: 每个邮件转换之后的维度有 10000 维
    feature_names = transfer.get_feature_names_out()
    pickle.dump(feature_names, open('data/03-模型训练特征.pkl', 'wb'), 3)


if __name__ == '__main__':
    # load_email_data()
    # clean_email_data()
    extract_email_feature()