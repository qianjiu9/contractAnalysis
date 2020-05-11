from keras.callbacks import ModelCheckpoint
import os
import numpy as np


def train(X_Y, model, get_model_checkpoint_path, data_generator, batch_size=32,
          epochs=1):
    """

    :param X_Y: [标签,训练文本]
    :param model: 待训练的模型
    :param get_model_checkpoint_path:获取新旧模型地址
    :param data_generator:数据迭代器
    :param batch_size:每轮次数据大小
    :param epochs:迭代次数
    :return:训练完成的模型
    """
    old_model_checkpoint_path, new_model_checkpoint_path = get_model_checkpoint_path()
    checkpoint = ModelCheckpoint(filepath=new_model_checkpoint_path, save_weights_only=True, save_best_only=True)

    if os.path.exists(old_model_checkpoint_path):
        model.load_weights(old_model_checkpoint_path)
        print('checkpoint load')

    total = len(X_Y)
    num_train = int(total * 0.90)
    num_test = total - num_train
    shuffle_indices = np.random.permutation(total)
    train_data = np.array(X_Y)[shuffle_indices[:num_train]]
    test_data = np.array(X_Y)[shuffle_indices[-num_test:]]
    print("train start")
    model.fit_generator(data_generator(train_data, batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(test_data, batch_size),
                        validation_steps=max(1, num_test // batch_size),
                        epochs=epochs,
                        initial_epoch=0,
                        callbacks=[checkpoint])
    print('train finish')
    return model, test_data


from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def evaluate(y_true, y_pre):
    tmp_shape = np.array(y_true)
    if len(tmp_shape) > 1:
        y_true = [np.argmax(y) for y in y_true]
    tmp_shape = np.array(y_pre)
    if len(tmp_shape) > 1:
        y_pre = [np.argmax(y) for y in y_pre]
    c = confusion_matrix(y_true, y_pre)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_pre)
    return c, precision, recall, fbeta_score, support
