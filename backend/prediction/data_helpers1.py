import re
import os
import numpy as np
import time
import datetime
import pickle
import jieba
from docx import Document
from keras.preprocessing.text import Tokenizer
import shutil
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.metrics import confusion_matrix

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

def get_chinese_stop_words(file_name):
    """
    加载中文停用词
    :param file_name: 中文停用词目录
    :return: 中文停用词 []
    """
    stop_words = []
    if file_name is None:
        return stop_words
    with open(file_name, encoding='utf-8') as f:
        for line in f.readlines():
            stop_words.append(line.strip())
    return stop_words


def clean_chars(string):
    """
    只保留中文字符串，删去其他字符，保留的各个中文字符串之间用空格间隔
    :param string:
    :param TREC:
    :return:
    """
    if string is None:
        return string
    pattern = re.compile(u"[\u4e00-\u9fa5]+")
    chinese_strs = re.findall(pattern, string)
    result = ""
    for sentense in chinese_strs:
        result = result + sentense
    return result


def clean_word(words_cut_generator, stop_words=None):
    """
    清理停用词
    :param word_list: 词序列
    :param stop_words: 停用词表
    :return: 去除停用词后的词序列
    """
    words = []
    for word in words_cut_generator:
        if (stop_words is None) or word not in stop_words:
            if word != ' ':
                words.append(word)
    return words


def get_pre_embedding(file, dim=300):
    """  分割线  """
    try:
        cache = get_cache_word_vectors(file)
        print("Loaded word embeddings from cache.")
        return cache
    except OSError:
        print("Didn't find embeddings cache file {}".format(file))

    # create the necessary dictionaries and the word embeddings matrix
    if os.path.exists(file):
        print('Indexing file {} ...'.format(file))

        word2idx = {}  # dictionary of words to ids
        idx2word = {}  # dictionary of ids to words
        embeddings = []  # the word embeddings matrix
        word_embeddings = {}  # dictionary of word to embedding

        # create the 2D array, which will be used for initializing
        # the Embedding layer of a NN.
        # We reserve the first row (idx=0), as the word embedding,
        # which will be used for zero padding (word with id = 0).
        embeddings.append(np.zeros(dim))

        # flag indicating whether the first row of the embeddings file
        # has a header
        header = False

        # read file, line by line
        with open(file, "r", encoding="utf-8") as f:
            index = 1
            for i, line in enumerate(f, 1):

                # skip the first row if it is a header
                if i == 1:
                    if len(line.split()) < dim:
                        header = True
                        continue

                values = line.strip().split(" ")
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')

                if word2idx.get(word) is None:
                    print(word)
                    idx2word[index] = word
                    word2idx[word] = index
                    embeddings.append(vector)
                    word_embeddings[word] = vector
                    index += 1

            # add an unk token, for OOV words
            if "<unk>" not in word2idx:
                idx2word[len(idx2word) + 1] = "<unk>"
                word2idx["<unk>"] = len(word2idx) + 1
                embeddings.append(
                    np.random.uniform(low=-0.05, high=0.05, size=dim))

            print(set([len(x) for x in embeddings]))

            print('Found %s word vectors.' % len(embeddings))
            embeddings = np.array(embeddings, dtype='float32')

        # write the data to a cache file
        save_cache_word_vectors(file, (word2idx, idx2word, embeddings, word_embeddings))

        return word2idx, idx2word, embeddings, word_embeddings


def save_cache_word_vectors(file, data):
    with open(get_file_cache_name(file), 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def get_cache_word_vectors(file):
    with open(get_file_cache_name(file), 'rb') as f:
        return pickle.load(f)


def get_file_cache_name(file):
    file_root, file_name = os.path.split(file)
    return os.path.join(file_root, file_name + "_cache")


def get_text_no_label(file):
    def load_docx(file):
        try:
            document = Document(file)
            text_str = ''
            for paragraph in document.paragraphs:
                text_str = text_str + '\n' + paragraph.text
            # print(text_str)
            return text_str
        except Exception:
            print('can not read file:' + str(file))
            return None

    type = get_file_type(file)
    if type == 'docx' or type == 'doc':
        return load_docx(file)
    with open(file, encoding='utf-8') as f:
        s = f.read()
        return s


def get_data_and_labels(file_dir, class_name_list, per_max_num=1500, sample_shuffle=False):
    """
    数据文件格式：file_dir是数据的根目录，每个类别数据放在同一个文件夹（类别名称为文件夹名）中。每条数据都是单独的一个文件
    :param file_dir: 文件所在根目录
    :param class_name_list: 分类名称
    :param per_max_num: 每个类别最大抽样数
    :param sample_shuffle: 随机抽样
    :return:data [[]],label []从0开始
    """
    data = []
    label = []
    filenames = []
    for index, class_name in enumerate(class_name_list):
        class_file_dir = os.path.join(file_dir, class_name).replace('\\', '/')
        count = 0
        file_list = os.listdir(class_file_dir)
        if sample_shuffle is True:
            shuffle = np.random.permutation(np.arange(len(file_list)))
            file_list = np.array(file_list)[shuffle]
        for i, file_name in enumerate(file_list):
            content = get_text_no_label(os.path.join(class_file_dir, file_name))
            if content is None:
                continue
            data.append(content)
            label.append(class_name)
            filenames.append(file_name)
            count = count + 1
            if count >= per_max_num:
                break
        print('load ' + class_name + ' ' + str(count))
    return data, label, filenames


def get_data_and_labels_and_name(file_dir, class_name_list, per_max_num=50, sample_shuffle=False):
    """
    数据文件格式：file_dir是数据的根目录，每个类别数据放在同一个文件夹（类别名称为文件夹名）中。每条数据都是单独的一个文件
    :param file_dir: 文件所在根目录
    :param class_name_list: 分类名称
    :param per_max_num: 每个类别最大抽样数
    :param sample_shuffle: 随机抽样
    :return:data [[]],label []从0开始
    """
    data = []
    label = []
    names = []
    for index, class_name in enumerate(class_name_list):
        class_file_dir = os.path.join(file_dir, class_name).replace('\\', '/')
        count = 0
        file_list = os.listdir(class_file_dir)
        if sample_shuffle is True:
            shuffle = np.random.permutation(np.arange(len(file_list)))
            file_list = np.array(file_list)[shuffle]
        for i, file_name in enumerate(file_list):
            try:
                text = get_text_no_label(os.path.join(class_file_dir, file_name))
                if text is None or len(text) <= 1:
                    continue
                data.append(text)
                label.append(class_name)
                names.append(file_name)
                count = count + 1
                if count >= per_max_num:
                    break
            except Exception as e:
                print(e)
                continue
        print('load ' + class_name + ' ' + str(count))
    return data, label, names


def get_file_type(file):
    strs = file.split('.')
    return strs[len(strs) - 1]


def get_idf(cuted_data):
    w_c = {}
    for text in cuted_data:
        w = {}
        for word in text:
            w[word] = w.get(word, 0) + 1  # 统计在一个文本中，每个词语出现次数
        for w, _ in w.items():
            w_c[w] = w_c.get(w, 0) + 1  # 统计包含该词语的文本数目
    w_cf = {}
    total = len(cuted_data)
    for w, c in w_c.items():
        f = np.log(total / c)  # 计算idf：对于某一个词语=总文本数目/包含该词语的文本数目，再取对数
        if len(w) == 1:  # 如果词语是单字，则放弃掉，因此令f=0
            f = 0
        w_cf[w] = (c, f)
    return w_cf


def get_class_tf(cuted_data):
    """
    获取类别词语的频率：即词语在给定类别中出现的频率/类别中词语总数
    :param cuted_data:[[]],每行表示一个文本，每个文本都经过分词处理
    :param top_n: 获取排名靠前的n个
    :return:
    """
    word_count = {}
    for words in cuted_data:
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1  # 统计每个词语出现的次数

    total = sum([c for w, c in word_count.items()])
    word_tf = {}
    for index, wc in enumerate(word_count.items()):
        word_tf[wc[0]] = (wc[1] / total)
    return word_count


def get_tf_idf(word_tf, word_idf, top_n=None):
    """
    计算tf_idf值
    :param word_tf:
    :param word_idf:
    :param top_n:
    :return:
    """
    word_tf_idf = {}
    for word, tf in word_tf.items():
        idf = word_idf.get(word, 0)
        word_tf_idf[word] = tf * idf

    sorted_word_tf_idf = sorted(word_tf_idf.items(), key=lambda wti: wti[1], reverse=True)  # 按照值进行排序
    res_dict = {}
    for index, value in enumerate(sorted_word_tf_idf):
        if top_n is not None and index > top_n:
            break
        res_dict[value[0]] = value[1]
    return res_dict


def process_data_common(documents, stop_words_file='baidu_stop_words.txt'):
    """
    分词
    去停用词
    #去低频词,暂时不作这个处理试试，原因是，有可能关键词语它就出现一次的
    :param documents: 文档列表[[]]
    :return:
    """
    stop_words = get_chinese_stop_words(stop_words_file)
    documents_cut_mat = []
    for index, text in enumerate(documents):
        text_cut = clean_word(jieba.cut(clean_chars(text)), stop_words)
        documents_cut_mat.append(text_cut)
        # if index % 100 == 0:
        #     print('has process ' + str(index))
    return documents_cut_mat


def get_model_cache(name):
    if not os.path.exists(get_cache_lda_name(name)):
        return []
    with open(get_cache_lda_name(name), 'rb') as f:
        print('get cache model : ' + str(name))
        return pickle.load(f)


def get_paragraph(document):
    paragraphs = document.split('\n')
    return paragraphs


def get_sentence(document):
    sentences = re.split('。|\.|\n', document)
    return sentences


def save_model_cache(data, name):
    print('save cache model : ' + str(name))
    with open(get_cache_lda_name(name), 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def get_cache_lda_name(name):
    return os.path.join(os.curdir, name + '_cache')


def process_data_4_cnn(cuted_data, labels, class_top_n=None):
    """
    1、统计idf
    2、根据label分成多组
    3、统计每组词语的tf
    4、综合词语形成词表
    5、在词表中的词语留下，不在词表中的直接去除
    :param cuted_data: 每行表示一个文本，每个文本都经过分词处理
    :param labels: 每个文本对应的类别
    :param class_top_n: 每类取排名靠前top_n个
    :return: 词表，处理后的数据矩阵，标签
    """
    word2idf = get_idf(cuted_data)  # the dictionary of word to idf

    # total types in labels
    label_set = set(labels)

    word_set = set()  # the set of word
    for label in label_set:
        temp_cut_datas = []
        for index, ori_label in enumerate(labels):
            if ori_label == label:
                temp_cut_datas.append(cuted_data[index])  # find data which has the same label ,and make them as a list
        word2classtf = get_class_tf(temp_cut_datas)  # the dictionary of word to class tf
        word2classtfidf = get_tf_idf(word2classtf, word2idf, class_top_n)  # thd dictionary of word to class word tf_idf
        word_set = word_set | word2classtfidf.keys()  # merge two set and remove duplicate element

    # remove word which not in word set
    res_data = []
    for data in cuted_data:
        temp_data = []
        for word in data:
            if word in word_set:
                temp_data.append(word)
        res_data.append(temp_data)

    # change set to dict, and the first index is 1
    tokenizer = Tokenizer(len(word_set))
    tokenizer.fit_on_texts(word_set)
    res_word_dict = tokenizer.word_index

    return res_word_dict, res_data, labels


def process_data_4_cnn_tfidf(cuted_data, labels, class_top_n=None):
    """
    1、统计idf
    2、根据label分成多组
    3、统计词语的tf
    4、综合词语形成词表
    5、在词表中的词语留下，不在词表中的直接去除
    :param cuted_data: 每行表示一个文本，每个文本都经过分词处理
    :param labels: 每个文本对应的类别
    :param class_top_n: 每类取排名靠前top_n个
    :return: 词表，处理后的数据矩阵，标签
    """
    word2idf = get_idf(cuted_data)  # the dictionary of word to idf

    # find data which has the same label ,and make them as a list
    word2classtf = get_class_tf(cuted_data)  # the dictionary of word to class tf
    tfidf = get_tf_idf(word2classtf, word2idf, class_top_n)  # thd dictionary of word to class word tf_idf
    # merge two set and remove duplicate element
    word_set = tfidf.keys()
    # remove word which not in word set
    res_data = []
    for data in cuted_data:
        temp_data = []
        for word in data:
            if word in word_set:
                temp_data.append(word)
        res_data.append(temp_data)

    # change set to dict, and the first index is 1
    tokenizer = Tokenizer(len(word_set))
    tokenizer.fit_on_texts(word_set)
    res_word_dict = tokenizer.word_index

    return res_word_dict, res_data, labels


def tokenize_data(datas, words_dict, text_length=600):
    """
    将所有的数据转换为标记
    :param words:
    :param datas:
    :param text_length: 最大长度，超过则截断
    :return:
    """
    res_data = []
    for data in datas:  # tokenize data
        token_data = [words_dict.get(w, 0) for w in data]
        if len(token_data) >= text_length:
            res_data.append(token_data[:text_length])
        else:
            for i in range(text_length - len(token_data)):
                token_data.append(0)
            res_data.append(token_data)
    return res_data


def tokenize_label(labels):
    """
    将标签也转换为标记
    :param labels:
    :return:
    """
    labels_set = set(labels)
    labels_dict = {}
    label_str = []
    sorted_set = list(labels_set)
    sorted_set = sorted(sorted_set)
    for index, label in enumerate(sorted_set):
        labels_dict[label] = index
        label_str.append(label)

    res_labels = []
    for label in labels:
        res_labels.append(labels_dict.get(label))

    return res_labels, label_str


def copy_to(src, dst, name):
    if not os.path.exists(src):
        return False
    if not os.path.exists(dst):
        os.makedirs(dst)
    src_path = os.path.join(src, name)
    dst_path = os.path.join(dst, name)
    try:
        shutil.copy(src_path, dst_path)
    except FileNotFoundError as fne:
        print(src_path + " not exist")
    return True


def cut_to(src, dst, name):
    success = copy_to(src, dst, name)
    if success:
        src_path = os.path.join(src, name)
        try:
            os.remove(src_path)
        except FileNotFoundError as fne:
            print(src_path + " not exist")
        finally:
            return False
        return True
    return False


def get_embedding(word_embedding, word_dict):
    sorted_word_list = sorted(word_dict.items(), key=lambda word2indx: word2indx[1])
    dim = len(list(word_embedding.values())[0])
    zero_vector = np.zeros([dim])
    res_embedding = []
    res_embedding.append(zero_vector)
    for index, word2idx in enumerate(sorted_word_list):
        word = word2idx[0]
        idx = word2idx[1]
        vector = word_embedding.get(word, np.random.rand(dim))
        if index + 1 != idx:
            print('error,word dict has some wrong for ' + word + ' index:' + str(index) + ' idx:' + str(idx))
        res_embedding.append(vector)
    return res_embedding


def split_2(data, shuffle=None, ratio=0.9):
    length = len(data)
    if shuffle is None:
        shuffle = np.random.permutation(length)
    data_1 = np.array(data)[shuffle[:int(length * ratio)]]
    data_2 = np.array(data)[shuffle[int(length * ratio):]]
    return data_1, data_2, shuffle


def print_result(y_true, y_pred, labels):
    # 混淆矩阵
    c = confusion_matrix(y_true, y_pred)
    # print(c)
    pds = pd.DataFrame(c, index=labels, columns=labels)

    print(pds)

    evals = eval_model(y_true, y_pred, labels)
    print(evals)
    return evals


def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': ['总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [999]
    res = pd.concat([res1, res2])
    return res[['Label', 'Precision', 'Recall', 'F1', 'Support']]


def save_eval_result(evals, out_path, cost, desc=None):
    path = out_path
    if not os.path.exists(path):
        fw = open(path, encoding='utf-8', mode='w+')
        fw.close()
    print('save eval result to ' + path)
    evals.to_csv(path, sep='\t', index=False, float_format='%.6f', mode='a')
    fw = open(path, encoding='utf-8', mode='a')
    fw.write('\n')
    fw.write('cost time: ' + str(cost) + 's')
    fw.write('\n')
    if desc is not None:
        fw.write('description is :' + desc)
        fw.write('\n')
    fw.write('\n')


def concatenate_str(words, tag=' '):
    res = ''
    try:
        for word in words:
            res = res + word + tag
    except Exception:
        print('???')
    return res


if __name__ == '__main__':
    t = datetime.datetime
    print('start:' + str(t.now()))
    # word2idx, idx2word, embeddings, word_embeddings = load_pre_embedding('sgns.sogou.word')
    text = clean_word(
        jieba.cut(clean_chars(
            get_text_no_label(os.path.join('..', 'Contract', 'rent_house_contract', '2015北京房屋租赁合同.docx')))))
    print('end:' + str(t.now()))
    # 分词
    # 去停用词
    # 去低频词

    # 针对ours的文本处理
