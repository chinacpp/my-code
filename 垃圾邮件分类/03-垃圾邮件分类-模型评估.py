import re
import joblib
import joblib
import zhconv
import jieba.posseg as psg
import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# 修改工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))


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


def evaluate():

    # 加载测试集数据
    test_data = pd.read_csv('data/01-原始邮件数据-测试集.csv')
    # 特征提取器
    vocab = pickle.load(open('data/03-模型训练特征.pkl', 'rb'))
    transfer = CountVectorizer(vocabulary=vocab)
    # 加载朴素贝叶斯垃圾邮件分类模型
    model = joblib.load('data/04-邮件分类模型.pth')

    # 测试集预测
    y_pred = []  # 存储测试集所有的预测结果
    for email in test_data['emails'].to_numpy():

        # 1. 数据清洗
        email = clean_data(email)
        # 2. 特征提取, 该函数要求传入的是二维列表
        email = transfer.transform([email]).toarray().tolist()
        # 3. 模型预测
        output = model.predict(email)
        # 4. 存储结果
        y_pred.append(output[0])

    print(len(y_pred), y_pred)

    # 测试集的真实标签
    y_true = test_data['labels'].tolist()

    # 对模型的预测结果进行不同指标的计算
    # 指标: 准确率、精度(垃圾邮件)、召回率(垃圾邮件)

    # 计算模型的准确率
    accuracy = accuracy_score(y_true, y_pred)
    print('accuracy: %.2f' % accuracy)
    # 计算模型对垃圾邮件预测精度
    precision = precision_score(y_true, y_pred, pos_label='spam')
    print('precision: %.2f' % precision)
    # 计算模型对垃圾邮件召回率
    recall = recall_score(y_true, y_pred, pos_label='spam')
    print('recall: %.2f' % recall)

    # 存储评估结果
    eval_result = {
        'sample_num': len(y_true),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

    pickle.dump(eval_result, open('data/06-模型评估结果.pkl', 'wb'), 3)


def predict(email):

    # 1. 加载特征提取器
    vocab = pickle.load(open('data/03-模型训练特征.pkl', 'rb'))
    transfer = CountVectorizer(vocabulary=vocab)
    # 2. 加载模型
    model = joblib.load('data/04-邮件分类模型.pth')
    # 3. 数据清洗
    email = clean_data(email)
    # 4. 特征提取
    email = transfer.transform([email]).toarray().tolist()
    # 5. 模型计算
    outputs = model.predict(email)

    return outputs[0]



if __name__ == '__main__':
    # evaluate()

    email = '''
    Received: from tom.com ([59.107.0.1])
	by spam-gw.ccert.edu.cn (MIMEDefang) with ESMTP id j84KXlix012575
	for <chi@ccert.edu.cn>; Tue, 6 Sep 2005 05:29:02 +0800 (CST)
Message-ID: <200509050433.j84KXlix012575@spam-gw.ccert.edu.cn>
From: =?GB2312?B?uePW3cfsyPA=?= <zhou@tom.com>
Subject: =?gb2312?B?S1ItUEPQzc7Cyqq2yNKjv9jPtc2z?=
To: chi@ccert.edu.cn
Content-Type: text/plain;charset="GB2312"
Reply-To: zhou@126.com
Date: Tue, 6 Sep 2005 05:48:50 +0800
X-Priority: 3
X-Mailer: Microsoft Outlook Express 6.00.2600.0000


KR-PC和G型温湿度遥控系统是广州庆瑞电子科技有限公司自行研制开发的高科技产品。
该产品采用计算机技术对环境温度、湿度进行远程自动多点检测和控制，
本系统广泛应用于高低压配电房、计算机房、仓库等需要对环境温、湿度有要求的场合。
KR-PC型温湿度遥控系统与除湿设备和冷却设备等各种设备配套使用，
用于高低压配电房、计算机房、仓库等需要对环境温度湿度有要求的场合，
起到自动防潮、除湿、降温等作用。当环境温度指示超过温度设定值上限时它能自动接通冷却设备电源，
当温度降低到温度设定下限时切断冷却设备电源，保证理想的环境温度；
当环境湿度指示超过湿度设定值上限时它能自动接通除湿器电源，
使环境湿度降低，当湿度下降到低于湿度设定值下限时，湿控器自动切断除湿器电源，
保证理想的环境湿度。当然也可根据客户的要求做成加热和加湿控制模式（默认出厂设置为降温和除湿模式）。
KR-PC型温湿度遥控系统是以功能很强的单片机为主的智能型仪器，
所有的子表串联到个人电脑上的RS232接口（即所谓的串口），子表本身可以显示和设置温湿度值，
个人电脑上也可以通过软件对子表进行数据采集和发送指令，显示和控制子表各种参数，
起到集中显示和控制的目的。
    '''
    
    result = predict(email)
    print('预测的类别:', result)

