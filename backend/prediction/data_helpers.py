import jieba
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np


def __concentrate__(word_generator):
    """
    将分词结果连接，以空格隔开
    :param word_generator: jieba分词后返回的词语生成器
    :return: 单个对象，各个词语中间由空格隔开
    """
    res = ''
    for word in word_generator:
        if word is '':
            continue
        res += word + ' '
    return res


class MyTokenizerForCNN():
    """
    训练文本的格式转换器，将以空格分开的文本，转换为索引表示，一次转换一条文本
    """

    def __init__(self, msgs, max_len, total_label):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(msgs)
        self.max_len = max_len
        self.total_label_dict = {l: i for i, l in enumerate(total_label)}
        self.output_size = len(total_label)
        self.vocab_size = len(self.tokenizer.word_index)

    def encode_text(self, text, max_len=None):
        token_data = pad_sequences(self.tokenizer.texts_to_sequences([text]), self.max_len, padding='post')
        return token_data[0]

    def encode_label(self, label):
        vec = [0] * self.output_size
        vec[self.total_label_dict.get(label)] = 1
        return vec


def data_generator_cnn(datas, batch_size):
    """
    数据迭代器，需要注意配合MyTokenizerForCNN以及processForCNN
    :param datas: [[文本在字典中的索引表示],[标签的one-hot表示]]
    :param batch_size:每批训练大小
    :return:
    """
    while True:
        shuffle_indices = np.random.permutation(len(datas))
        X, Y = [], []
        for i in shuffle_indices:
            x = datas[i][0]
            y = datas[i][1]
            Y.append(y)
            X.append(x)
            if len(X) >= batch_size or i == shuffle_indices[-1]:
                yield np.array(X), np.array(Y)
                [X, Y] = [], []


def processForCNN(just_review_datas, input_size, tokenizer=None):
    """
    :param just_review_datas: [label,msg]
    :return:
    """
    cuted_datas = [[label, jieba.cut(msg)] for label, msg in just_review_datas]  # 分词
    train_datas = [[label, __concentrate__(msg)] for label, msg in cuted_datas]  # 连接词语，便于生成tokenizer
    train_labels = [label for label, _ in train_datas]

    if tokenizer is None:
        tmp_labels = list(set(train_labels))
        tmp_labels = list(sorted(tmp_labels))
        tmp_msgs = [msg for l, msg in train_datas]
        tokenizer = MyTokenizerForCNN(tmp_msgs, input_size, tmp_labels)

    token_train_datas = [tokenizer.encode_text(msg) for label, msg in train_datas]  # 将训练数据转换为token表示
    token_train_labels = [tokenizer.encode_label(label) for label in train_labels]  # 将标签转换为one-hot格式

    return token_train_datas, token_train_labels, tokenizer
