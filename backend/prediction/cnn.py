from keras import *
from keras.layers import *
from keras.optimizers import Adam


class CNN():
    def __init__(self, input_size, output_size, vocab_size, embed_size, filter_sizes, filters_num, learn_rate):
        """

        :param input_size: 输入文本最大长度
        :param output_size: 类别总数
        :param vocab_size: 词表总数
        :param embed_size: 词向量维度
        :param filter_sizes: 卷积核大小
        :param filters_num: 卷积核数量
        :param learn_rate: 学习率
        """
        input = Input(shape=[input_size])

        x = Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=input_size, trainable=True)(input)
        convs = []
        for kernel_size in filter_sizes:
            conv = Conv1D(filters=filters_num, kernel_size=kernel_size)(x)
            pool = MaxPooling1D(input_size - kernel_size + 1)(conv)
            convs.append(pool)
        conv_contact = concatenate(convs, axis=-1)
        conv_contact = Flatten()(conv_contact)

        x = Dropout(0.5)(conv_contact)
        x = Dense(units=output_size)(x)
        output = Activation(activations.sigmoid, name='out')(x)

        model = Model(input, output)

        model.compile(loss=losses.categorical_crossentropy, optimizer=Adam(learn_rate), metrics=['accuracy'])
        model.summary()
        self.model = model
