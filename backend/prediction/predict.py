import jieba
from backend.prediction.data_helpers1 import clean_chars, __concentrate__
from backend.prediction.cnn import CNN
import numpy as np
import os
import _pickle as c_pickle
import keras


input_size = 900
output_size = 5
filter_sizes = [3, 4, 6]
filter_num = 128
learn_rate = 1e-3
embed_size = 300

typelist = ["销售合同", "借款合同", "劳务合同", "项目合同", "合作合同", "其他合同"]
base_dir = os.path.dirname(__file__)
path = os.path.join(base_dir, 'data.pkl')

checkpoint_path = './backend/prediction/runs/model_3_12.h5'
f1 = open(path, 'rb')
tokenizer = c_pickle.load(f1)
def predict_type(text):
    print('predicting')
    keras.backend.clear_session()
    cut_text = jieba.cut(clean_chars(text))
    msg = __concentrate__(cut_text)
    encode_msg = tokenizer.encode_text(msg)
    cnn = CNN(input_size=input_size,
              output_size=output_size,
              vocab_size=tokenizer.vocab_size + 1,
              embed_size=embed_size,
              filter_sizes=filter_sizes,
              filters_num=filter_num,
              learn_rate=learn_rate)
    if os.path.exists(checkpoint_path):
        cnn.model.load_weights(checkpoint_path)

    res = cnn.model.predict(np.array([encode_msg]))
    res_end = res.tolist()[0].index(max(res.tolist()[0]))
    rate = res[0][res_end]
    if rate<0.3:
        res_end = 5
    print('预测概率：')
    print(res[0][res_end])
    return typelist[res_end]
