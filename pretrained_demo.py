import csv
import tensorflow as tfv2
from tqdm import tqdm

import numpy as np
import pandas as pd

from model import single_model

import tensorflow_addons as tfa


from transformers import TFBertModel, BertTokenizer
huggingface_tag = '../huggingface/Bert/bert-base-chinese'

class DataManager:
    def __init__(self,max_seq_length,tokenizer):
        self.max_sequence_length = max_seq_length
        self.tokenizer = tokenizer
        self.UNKNOWN = '[UNK]'
        self.PADDING = '[PAD]'
        # 读取不了非CSV文件
        self.train_file = 'data/people/train.csv'
        self.dev_file = 'data/people/dev.csv'
        self.label2id_file = 'data/people/label2id.txt'

        # 构建词表，如果是bert，只需要label2id和id2label
        self.token2id, self.id2token, self.label2id, self.id2label = self.build_vocab(self.train_file)

    # 读取训练集、验证集，经过预训练模型的tokenizer，并shuffle后的结果
    def get_train_dev_data(self):
        # 1. 构建训练集
        df_train = pd.read_csv(self.train_file, sep=" ", quoting=csv.QUOTE_NONE,
                               skip_blank_lines=False, header=None, names=['token', 'label'])
        # 这里调用tokenizer转换成向量 (if use_pretrained_model)
        X, y, att_mask = self.prepare_pretrained_embedding(df_train)
        # shuffle the samples
        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        att_mask = att_mask[indices]

        # 2.构建验证集
        X_train = X
        y_train = y
        att_mask_train = att_mask
        X_val, y_val, att_mask_val = self.get_dev_data()
        print('training set size: {}, validating set size: {}'.format(len(X_train), len(X_val)))
        # write('training set size: {}, validating set size: {}'.format(len(X_train), len(X_val)))
        train_dataset = tfv2.data.Dataset.from_tensor_slices((X_train, y_train, att_mask_train))
        val_dataset = tfv2.data.Dataset.from_tensor_slices((X_val, y_val, att_mask_val))

        return train_dataset, val_dataset

    # 读取验证集
    def get_dev_data(self):
        df_val = pd.read_csv(self.dev_file, sep=" ", quoting=csv.QUOTE_NONE,
                               skip_blank_lines=False, header=None, names=['token', 'label'])
        # if self.configs.use_pretrained_model:
        X_val, y_val, att_mask_val = self.prepare_pretrained_embedding(df_val)
        return X_val, y_val, att_mask_val

    # 根据BIO信息调用embedding tokenizer转换,label的转换
    def prepare_pretrained_embedding(self,df):
        """
        :param df:
        :return:
        """
        X = []
        y = []
        att_mask = []
        tmp_x = []
        tmp_y = []
        lines = df.token.isnull().sum()
        with tqdm(total=lines, desc='loading data') as bar:
            for _, record in df.iterrows():
                token = record.token
                label = record.label
                # 循环到了句子末尾时，处理
                if str(token) == str(np.nan):
                    if len(tmp_x) <= self.max_sequence_length - 2:
                        # 这里返回会多了 [CLS] 和 [SEP]，token_id分别是101，102
                        tmp_x = self.tokenizer.encode(tmp_x)
                        tmp_att_mask = [1] * len(tmp_x)
                        tmp_y = [self.label2id[y] for y in tmp_y]
                        #  [CLS] 和 [SEP]的label是'O'
                        tmp_y.insert(0, self.label2id['O'])
                        tmp_y.append(self.label2id['O'])
                        # [padding]的label_id是0
                        tmp_x += [0 for _ in range(self.max_sequence_length - len(tmp_x))]
                        tmp_y += [self.label2id[self.PADDING] for _ in range(self.max_sequence_length - len(tmp_y))]
                        tmp_att_mask += [0 for _ in range(self.max_sequence_length - len(tmp_att_mask))]
                        X.append(tmp_x)
                        y.append(tmp_y)
                        att_mask.append(tmp_att_mask)
                    else:
                        # 此处的padding不能在self.max_sequence_length加2，否则不同维度情况下，numpy没办法转换成矩阵
                        tmp_x = tmp_x[:self.max_sequence_length - 2]
                        tmp_x = self.tokenizer.encode(tmp_x)
                        X.append(tmp_x)
                        tmp_y = tmp_y[:self.max_sequence_length - 2]
                        tmp_y = [self.label2id[y] for y in tmp_y]
                        tmp_y.insert(0, self.label2id['O'])
                        tmp_y.append(self.label2id['O'])
                        y.append(tmp_y)
                        tmp_att_mask = [1] * self.max_sequence_length
                        att_mask.append(tmp_att_mask)
                    tmp_x = []
                    tmp_y = []
                    bar.update()
                    # todo 测试使用，加载少量数据
                    # break
                else:
                    tmp_x.append(token)
                    tmp_y.append(label)
        return np.array(X), np.array(y), np.array(att_mask)


    # 创建词表
    def build_vocab(self, train_path):
        """
        根据训练集生成词表
        :param train_path:
        :return:
        """
        df_train = pd.read_csv(train_path, sep=" ", quoting=csv.QUOTE_NONE,
                               skip_blank_lines=False, header=None, names=['token', 'label'])
        token2id, id2token = {}, {}

        # todo 传统embedding
        # if not self.configs.use_pretrained_model:
        #     tokens = list(set(df_train['token'][df_train['token'].notnull()]))
        #     # 过滤掉为空的token不纳入词表
        #     tokens = [tokens for token in tokens if token if token not in [' ', '']]
        #     token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        #     id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        #     id2token[0] = self.PADDING
        #     token2id[self.PADDING] = 0
        #     # 向生成的词表中加入[UNK]
        #     id2token[len(tokens) + 1] = self.UNKNOWN
        #     token2id[self.UNKNOWN] = len(tokens) + 1
        #     # 保存词表及标签表
        #     with open(self.token2id_file, 'w', encoding='utf-8') as outfile:
        #         for idx in id2token:
        #             outfile.write(id2token[idx] + '\t' + str(idx) + '\n')

        labels = list(set(df_train['label'][df_train['label'].notnull()]))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        id2label[0] = self.PADDING
        label2id[self.PADDING] = 0
        with open(self.label2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + '\t' + str(idx) + '\n')
        return token2id, id2token, label2id, id2label

# 测试主流程训练过程，bert+bilstm+crf
# todo 根据loss学习
def test_main_process():
    tokenizer = BertTokenizer.from_pretrained(huggingface_tag,from_pt=True)
    pretrained_model = TFBertModel.from_pretrained(huggingface_tag,from_pt=True)
    print(pretrained_model)

    dm = DataManager(max_seq_length=single_model.Setting().num_steps, tokenizer=tokenizer)

    train_dataset, val_dataset = dm.get_train_dev_data()
    batch_size = 32

    # num_classes是 len(label2id)
    num_classes = len(dm.label2id)
    my_transition_params = tfv2.Variable(tfv2.random.uniform(shape=(num_classes, num_classes)))

    for step,batch in train_dataset.shuffle(len(train_dataset)).batch(batch_size).enumerate():
        X_train_batch, y_train_batch, att_mask_batch = batch
        # todo （batch_size,max_len,不知道是什么??）
        embedding_inputs = pretrained_model(X_train_batch, attention_mask=att_mask_batch)[0]
        # 计算没有加入pad之前的句子的长度，因为padding的id是0。参数 axis=1 表示在每个序列（行）上统计非零元素的数量。
        # 后续的模型训练需要inputs_length
        inputs_length = tfv2.math.count_nonzero(X_train_batch, 1)

        mydropout = tfv2.keras.layers.Dropout(0.5)
        training = 1
        outputs = mydropout(embedding_inputs, training)

        lstm_hidden_dim = 200
        mybilstm = tfv2.keras.layers.Bidirectional(tfv2.keras.layers.LSTM(lstm_hidden_dim, return_sequences=True))
        outputs = mybilstm(outputs)

        mydense = tfv2.keras.layers.Dense(num_classes)
        logits = mydense(outputs)

        targets = y_train_batch
        tensor_targets = tfv2.convert_to_tensor(targets)

        log_likelihood, transition_params = tfa.text.crf.crf_log_likelihood(
            logits, tensor_targets, inputs_length, transition_params=my_transition_params)
        # log_likelihood是(batch_size,)，表示一个step（batch）下每个句子的crf得分
        print(log_likelihood)
        # 对单个批次下的所有crf得分求均值作为单个batch的损失
        loss = -tfv2.reduce_mean(log_likelihood)
        info = 'step = {}, loss = {}'.format(step, loss)
        print(info)







def test_tokenizer():
    from transformers import TFBertModel, BertTokenizer

    # 加载预训练模型和分词器
    huggingface_tag = '../huggingface/Bert/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(huggingface_tag)
    input = tokenizer.encode(['张', '三', '是'])
    print(input) # 这个返回值是5位，因为会把[cls]和[sep]给加上[cls]是101，[sep]是102
    input = tokenizer.encode(['张', '四', '我'])
    print(input)



# # 定义一个简单的分类头
# class NERModel(tf.keras.Model):
#     def __init__(self, pretrained_model, num_labels):
#         super(NERModel, self).__init__()
#         self.pretrained = pretrained_model
#         self.classifier = tf.keras.layers.Dense(num_labels, activation='softmax')
#
#     def call(self, inputs, attention_mask=None):
#         # 使用预训练模型提取特征
#         sequence_output = self.pretrained(inputs, attention_mask=attention_mask)[0]
#         # 通过分类头得到每个 token 的标签分布from_pt=True
#         logits = self.classifier(sequence_output)
#         return logits
#
# 假设有3个NER标签：B, I, O
# num_labels = 3
# # 构建 NER 模型
# ner_model = NERModel(pretrained_model, num_labels)
#
# # 编译模型
# ner_model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=['accuracy']
# )
#
# # 假设 batch 是 (X_train_batch, y_train_batch, att_mask_batch)
# # X_train_batch: 输入的 token id
# # y_train_batch: 对应的 NER 标签
# # att_mask_batch: attention mask
#
# # 模拟一个 batch 的训练过程
# X_train_batch = tf.constant([[101, 2769, 3377, 784, 102]])  # 示例输入
# y_train_batch = tf.constant([[0, 1, 1, 2, 0]])              # 示例标签
# att_mask_batch = tf.constant([[1, 1, 1, 1, 1]])              # 示例 mask
#
# # 前向传播
# logits = ner_model(X_train_batch, attention_mask=att_mask_batch)
#
# # 打印输出
# print("Logits:", logits)


def demo():
    from transformers import TFBertModel
    huggingface_tag = '../huggingface/Bert/bert-base-chinese'
    pretrained_model = TFBertModel.from_pretrained(huggingface_tag, from_pt=True)
    print(pretrained_model)
# def demo():
#     train_dataset, val_dataset = get_training_set()
#     pretrained_model = TFBertModel.from_pretrained(huggingface_tag, from_pt=True)
#     print(pretrained_model)
#
#     X_train_batch, y_train_batch, att_mask_batch = batch
#     model_inputs = pretrained_model(X_train_batch, attention_mask=att_mask_batch)[0]

if __name__ == '__main__':
    # demo()
    # test_tokenizer()
    test_main_process()