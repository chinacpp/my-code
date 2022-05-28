from sklearn.naive_bayes import MultinomialNB
import pickle
import joblib


def train():

    # 1. 加载训练数据
    train_data = pickle.load(open('data/03-模型训练数据.pkl', 'rb'))

    # 2. 模型的训练
    model = MultinomialNB()
    model.fit(train_data['emails'], train_data['labels'])

    # 3. 模型的存储
    joblib.dump(model, 'data/04-邮件分类模型.pth')

    # 5万+训练数据，每一条数据有 1 万特征


if __name__ == '__main__':
    train()
