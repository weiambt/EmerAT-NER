import csv
import tensorflow as tfv2
from tqdm import tqdm
from transformers import TFBertModel, BertTokenizer
import numpy as np
import pandas as pd

import model.mtl_model
from model import single_model
from model.Configure import Configure

# class DataSet:
#     def __init__(self,X,y,attention_mask):
#         self.X = X
#         self.y = y
#         self.attention_mask = attention_mask
#         # 数据集中句子个数
#         self.data_size = len(X)


class DataManager:
    def __init__(self,setting,dataset):
        self.UNKNOWN = '[UNK]'
        self.PADDING = '[PAD]'
        self.max_sequence_length = setting.num_steps
        self.huggingface_tag = setting.huggingface_tag

        # 读取不了非CSV文件
        self.train_file = dataset.train_file
        self.dev_file = dataset.dev_file
        self.label2id_file = dataset.label2id_file
        self.suffix = dataset.suffix

        self.tokenizer = BertTokenizer.from_pretrained(self.huggingface_tag)
        # 构建词表，如果是bert，只需要label2id和id2label
        self.token2id, self.id2token, self.label2id, self.id2label = self.build_vocab(self.train_file)
        self.max_label_number = len(self.label2id)

        # 读取训练集、验证集，经过预训练模型的tokenizer，并shuffle后的结果
    # return train_dataset,dev_dataset

    def get_train_dev_data(self,less_data_flag=False):
        # 1. 构建训练集
        df_train = pd.read_csv(self.train_file, sep=" ", quoting=csv.QUOTE_NONE,
                               skip_blank_lines=False, header=None, names=['token', 'label'])
        # 这里调用tokenizer转换成向量 (if use_pretrained_model)
        X, y, att_mask = self.prepare_pretrained_embedding(df_train,less_data_flag)
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
        X_val, y_val, att_mask_val = self.get_dev_data(less_data_flag)
        print('training set size: {}, validating set size: {}'.format(len(X_train), len(X_val)))
        # write('training set size: {}, validating set size: {}'.format(len(X_train), len(X_val)))
        train_dataset = tfv2.data.Dataset.from_tensor_slices((X_train, y_train, att_mask_train))
        val_dataset = tfv2.data.Dataset.from_tensor_slices((X_val, y_val, att_mask_val))

        # todo 因为不支持tf.dataset所以用Numpy返回
        # return DataSet(X_train,y_train,att_mask_train),DataSet(X_val,y_val,att_mask_val)
        return train_dataset, val_dataset

    # 读取验证集
    def get_dev_data(self,less_data):
        df_val = pd.read_csv(self.dev_file, sep=" ", quoting=csv.QUOTE_NONE,
                               skip_blank_lines=False, header=None, names=['token', 'label'])
        # if self.configs.use_pretrained_model:
        X_val, y_val, att_mask_val = self.prepare_pretrained_embedding(df_val,less_data)
        return X_val, y_val, att_mask_val

    # 根据BIO信息调用embedding tokenizer转换,label的转换
    def prepare_pretrained_embedding(self,df,less_data):

        X = []
        y = []
        att_mask = []
        tmp_x = []
        tmp_y = []
        lines = df.token.isnull().sum()
        cnt = 0
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
                    if less_data and cnt>500:
                        break
                    cnt += 1
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


def test_DataManger():

    train_file = '../data/people/train.csv'
    dev_file = '../data/people/dev.csv'
    label2id_file = '../data/people/label2id.txt'
    huggingface_tag = '../../huggingface/Bert/bert-base-chinese'

    cfg = Configure(train_file=train_file, dev_file=dev_file, label2id_file=label2id_file,huggingface_tag=huggingface_tag)
    setting = single_model.Setting()
    dm = DataManager(cfg,setting)

    train_dataset, val_dataset = dm.get_train_dev_data()

if __name__ == '__main__':
    test_DataManger()