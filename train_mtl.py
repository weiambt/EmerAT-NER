import math
import time
from logging import Logger

import numpy as np
import os
import tensorflow_addons as tfa

import tensorflow as tfv2
import tensorflow.compat.v1 as tf

from keras.optimizers import Adam
from tqdm import tqdm
from transformers import TFBertModel

from DataSet import DataSet
from model.DataManager import DataManager

import utils.metric as MetricUtil
import utils.tool as Tool
import conf.cfg

# todo 下面待注释？？
# tf.disable_eager_execution()
# tf.disable_v2_behavior()

from model import mtl_model
# from utils.common import CommonUtil
import utils.logger as logger

from tensorflow.keras import backend as K
K.clear_session()

class Setting(object):
    def __init__(self):
        # global parameter
        self.adv_weight = 0.06
        self.task_num = 2

        # dataset config
        self.dataset_src = DataSet("people")
        self.dataset_tgt = DataSet("emergency_2024_4_14")
        self.less_data_flag = False

        self.checkpoints_dir = './ckpt/ner-cws-2025-01-10'
        self.checkpoint_name = 'model'
        self.pretrained_model_name = 'bert-base-chinese'

        # self.huggingface_tag = '/Users/didi/Desktop/KYCode/huggingface/bert-base-chinese'
        # self.huggingface_tag = 'E:\\ARearchCode\\huggingface\\bert-base-chinese'

        self.is_early_stop = True
        self.patience = 50

        # common train parameter
        self.epoches = 100
        self.batch_size = 10
        # self.num_steps 通常表示输入序列的固定最大长度，不足的补padding，多的截断
        self.num_steps = 300
        self.lr = 0.001
        self.word_dim = 100
        self.lstm_dim = 120
        self.num_units = 240
        self.num_heads = 8
        self.max_to_keep = 3
        self.clip = 5
        self.in_keep_prob = 0.7
        self.out_keep_prob = 0.6

        # dataset  train
        self.keep_prob_src = 0.7
        self.keep_prob_tgt = 0.7

class TrainMtl:
    def __init__(self,setting,conf,logger):
        self.setting = setting
        self.logger = logger
        self.conf = conf
        self.setting.huggingface_tag = os.path.join(self.conf.huggingface_dir, self.setting.pretrained_model_name)

        # self.initializer = tfv2.keras.initializers.GlorotUniform()

        self.datamanager_src = DataManager(self.logger,self.setting, self.setting.dataset_src)
        self.datamanager_tgt = DataManager(self.logger,self.setting, self.setting.dataset_tgt)

        self.pretrained_model = TFBertModel.from_pretrained(self.setting.huggingface_tag, from_pt=True)
        self.ner_model = mtl_model.TransferModel(self.setting, self.datamanager_src,self.datamanager_tgt)

        self.optimizer = Adam(learning_rate=self.setting.lr)
        self.global_step_src = tfv2.Variable(0, name="global_step_src", trainable=False)
        self.global_step_tgt = tfv2.Variable(0, name="global_step_tgt", trainable=False)

        checkpoint = tf.train.Checkpoint(ner_model=self.ner_model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.setting.checkpoints_dir, checkpoint_name=self.setting.checkpoint_name,
            max_to_keep=self.setting.max_to_keep)

    def train(self):
        final_start_time = time.time()
        # self.embedding = tfv2.cast(np.load('./data/weibo_vector.npy'), tf.float32)

        train_dataset_src, dev_dataset_src = self.datamanager_src.get_train_dev_data(self.setting.less_data_flag)
        train_dataset_tgt, dev_dataset_tgt = self.datamanager_tgt.get_train_dev_data(self.setting.less_data_flag)

        self.task_label_tgt = [[1, 0] for _ in range(self.setting.batch_size)]
        self.task_label_src = [[0, 1] for _ in range(self.setting.batch_size)]

        best_f1_val = 0.0
        best_step = -1
        best_at_epoch = -1
        unprocessed = 0

        # todo 风险：如果一个数据集数据用完了，那么会执行结束，不会再执行多的数据集
        for epoch in range(self.setting.epoches):
            self.logger.info("------epoch {}\n".format(epoch))
            start_time = time.time()
            batches_src = train_dataset_src.shuffle(len(train_dataset_src)).batch(self.setting.batch_size,drop_remainder=True).enumerate()
            batches_tgt = train_dataset_tgt.shuffle(len(train_dataset_tgt)).batch(self.setting.batch_size,drop_remainder=True).enumerate()

            # 枚举到最小的数据批次：min(batch_size_src,batch_size_tgt)
            for step, batch_src,batch_tgt in zip(range(len(batches_tgt)), batches_src, batches_tgt):
                #  初始化计算图参数，第一次执行初始化所有参数
                if epoch == 0 and step == 0:
                    X_train_batch, y_train_batch, att_mask_batch = batch_src[1]
                    embedding_inputs = self.pretrained_model(X_train_batch,
                                                             attention_mask=att_mask_batch)[0]
                    inputs_length_bacth = tfv2.math.count_nonzero(X_train_batch, 1)


                    self.ner_model.multi_task_init(embedding_inputs,inputs_length_bacth,y_train_batch,
                                                                                is_train=True,
                                                                                task_label=self.task_label_src)

                # 枚举所有任务
                for task in ["src","tgt"]:
                    if task == "src":
                        is_target = False
                        task_label = self.task_label_src
                        X_train_batch, y_train_batch, att_mask_batch = batch_src[1]
                        # todo （batch_size,max_len,不知道是什么??）
                        embedding_inputs = self.pretrained_model(X_train_batch,
                                                                 attention_mask=att_mask_batch)[0]
                        inputs_length_bacth = tfv2.math.count_nonzero(X_train_batch, 1)
                    else:
                        is_target = True
                        task_label = self.task_label_tgt
                        X_train_batch, y_train_batch, att_mask_batch = batch_tgt[1]
                        # todo （batch_size,max_len,不知道是什么??）
                        embedding_inputs = self.pretrained_model(X_train_batch,
                                                                 attention_mask=att_mask_batch)[0]
                        # 普通嵌入
                        # embedding_inputs = tfv2.nn.embedding_lookup(self.embedding, X_train_batch)

                        # 计算没有加入pad之前的句子的长度，因为padding的id是0。参数 axis=1 表示在每个序列（行）上统计非零元素的数量。
                        # 后续的模型训练需要inputs_length
                        inputs_length_bacth = tfv2.math.count_nonzero(X_train_batch, 1)


                    with tfv2.GradientTape() as tape:
                        # 前向传播，计算损失
                        logits, transition_params, loss = self.ner_model.multi_task(embedding_inputs,
                                                                                    inputs_length_bacth,
                                                                                    y_train_batch,
                                                                                    is_train=True,
                                                                                    is_target=is_target,
                                                                                    task_label=task_label)
                        if task == "tgt":
                            current_step = int(self.global_step_tgt.numpy())
                            if current_step % 20 == 0:
                                self.logger.info("step {},loss {}\n".format(current_step, loss))

                    # 定义好参加梯度的参数
                    variables = self.ner_model.trainable_variables
                    # 将预训练模型里面的pooler层的参数去掉
                    variables = [var for var in variables if 'pooler' not in var.name]

                    # 计算梯度
                    gradients = tape.gradient(loss, variables)
                    # 如果需要梯度裁剪
                    if self.setting.clip > 0:
                        gradients, _ = tfv2.clip_by_global_norm(gradients, clip_norm=self.setting.clip)
                    # 反向传播，自动微分计算
                    self.optimizer.apply_gradients(zip(gradients, variables))
                    if task == "src":
                        self.global_step_src.assign_add(1)
                    else:
                        self.global_step_tgt.assign_add(1)

                    # val_f1_avg, val_res_str = self.validate(dev_dataset_tgt)
                    # return

            # 每个epoch后验证模型
            val_f1_avg, val_res_str = self.validate(dev_dataset_tgt)
            time_exec = (time.time() - start_time) / 60
            self.logger.info('epoch %d: time consumption:%.2f(min), %s\n' % (epoch,time_exec, val_res_str))

            if np.array(val_f1_avg).mean() > best_f1_val:
                best_f1_val = np.array(val_f1_avg).mean()
                best_at_epoch = epoch + 1
                self.checkpoint_manager.save()
                self.logger.info(('===== saved the new best model with f1: %.3f \n' % best_f1_val))
                unprocessed = 0
            else:
                unprocessed += 1

            if self.setting.is_early_stop:
                if unprocessed >= self.setting.patience:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(self.setting.patient))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                    self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - final_start_time) / 60))
                    return

        self.logger.info('========= final best f1 = {},on epoch {},on step {}'.format(best_f1_val, best_at_epoch, best_step))
        final_exec_time = (time.time() - final_start_time)/60
        self.logger.info('execute time is {} min\n,'.format(str(int(final_exec_time))))

    def validate(self, val_dataset):
        # validation
        num_val_iterations = int(math.ceil(1.0 * len(val_dataset) / self.setting.batch_size))
        self.logger.info('start evaluate engines...')
        loss_values = []
        val_results = {}
        val_labels_results = {}
        for label in self.datamanager_tgt.suffix:
            val_labels_results.setdefault(label, {})
        measuring_metrics = ["precision", "recall", "f1", "accuracy"]
        for measure in measuring_metrics:
            val_results[measure] = 0
        for label, content in val_labels_results.items():
            for measure in measuring_metrics:
                if measure != 'accuracy':
                    val_labels_results[label][measure] = 0

        for val_batch in tqdm(val_dataset.batch(self.setting.batch_size,drop_remainder=True)):
            X_val_batch, y_val_batch, att_mask_batch = val_batch

            embedding_inputs = self.pretrained_model(X_val_batch, attention_mask=att_mask_batch)[0]
            # 普通嵌入
            # embedding_inputs = tfv2.nn.embedding_lookup(self.embedding, X_val_batch)

            inputs_length_val = tf.math.count_nonzero(X_val_batch, 1)
            logits_val, transition_params_val, val_loss = self.ner_model.multi_task(embedding_inputs,
                                                                                    inputs_length_val,
                                                                                    y_val_batch,
                                                                                    is_train=False,
                                                                                    is_target=True,
                                                                                    task_label=self.task_label_tgt)
            batch_pred_sequence_val, _ = tfa.text.crf_decode(logits_val,
                                                             transition_params_val,
                                                             inputs_length_val)
            measures, lab_measures = MetricUtil.metrics(X_val_batch,
                                                        y_val_batch,
                                                        batch_pred_sequence_val,
                                                        self.datamanager_tgt)

            for k, v in measures.items():
                val_results[k] += v
            for lab in lab_measures:
                if lab not in val_labels_results:
                    val_labels_results.setdefault(lab, {})
                    for measure in measuring_metrics:
                        if measure != 'accuracy':
                            val_labels_results[lab][measure] = 0
                for k, v in lab_measures[lab].items():
                    val_labels_results[lab][k] += v
            loss_values.append(val_loss)

        val_res_str = ''
        val_f1_avg = 0
        for k, v in val_results.items():
            val_results[k] /= num_val_iterations
            val_res_str += (k + ': %.3f ' % val_results[k])
            if k == 'f1':
                val_f1_avg = val_results[k]
        for label, content in val_labels_results.items():
            val_label_str = ''
            for k, v in content.items():
                val_labels_results[label][k] /= num_val_iterations
                val_label_str += (k + ': %.3f ' % val_labels_results[label][k])
            info = ('label: %s, %s' % (label, val_label_str))
            self.logger.info(info)
        return val_f1_avg, val_res_str

        # 词表embedding
        # self.embedding = None


def test():
    a = [1, 2, 3]
    b = [4, 5]

    for x,y in zip(a, b):
        print(x,y)
    for x,y in zip(b, a):
        print(x,y)
    for idx,x,y in zip(range(len(a)),a, b):
        print(idx,x,y)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tfv2.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tfv2.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    conf = conf.cfg.get_cfg_by_os()
    logger = logger.get_logger('./log')
    setting = Setting()
    if not os.path.exists(setting.checkpoints_dir):
        os.makedirs(setting.checkpoints_dir)
    logger.info("================Setting===================")
    Tool.print_vars(logger,setting)
    TrainMtl(setting,conf,logger).train()