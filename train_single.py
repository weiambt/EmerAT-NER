import math
import time
import numpy as np
import os
import tensorflow_addons as tfa

import tensorflow as tfv2
import tensorflow.compat.v1 as tf
from keras.optimizers import Adam
from tqdm import tqdm
from transformers import TFBertModel

from model.Configure import Configure
from model.DataManager import DataManager

# tf.disable_eager_execution()
# tf.disable_v2_behavior()

from model import single_model
from utils.common import CommonUtil
import utils.metric as MetricUtil


class TrainSingle:
    def __init__(self):
        train_file = 'data/people/train.csv'
        dev_file = 'data/people/dev.csv'
        label2id_file = 'data/people/label2id.txt'
        suffix = ["ORG", "PER", "LOC"]

        huggingface_tag = '/Users/didi/Desktop/KYCode/huggingface/Bert/bert-base-chinese'
        self.checkpoints_dir = './ckpt/ner-cws-2025-01-03'
        self.checkpoint_name = 'model'
        max_to_keep = 3
        # 词表embedding
        self.embedding = None


        self.initializer = tfv2.keras.initializers.GlorotUniform()
        self.setting = single_model.Setting()
        self.cfg = Configure(train_file=train_file, dev_file=dev_file, label2id_file=label2id_file,
                        huggingface_tag=huggingface_tag, suffix=suffix)
        self.datamanager = DataManager(self.cfg, self.setting)

        self.ner_model = single_model.Model(self.setting,self.datamanager.max_label_number)
        self.pretrained_model = TFBertModel.from_pretrained(huggingface_tag, from_pt=True)
        self.optimizer = Adam(learning_rate=0.001)
        self.global_step = tfv2.Variable(0, name="global_step", trainable=False)
        checkpoint = tf.train.Checkpoint(ner_model=self.ner_model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.checkpoints_dir, checkpoint_name=self.checkpoint_name, max_to_keep=max_to_keep)


    def train(self):
        # 如果不存在,创建文件
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        # model_name = 'model-{}'.format(dir_path)
        out = open('{}/log.txt'.format(self.checkpoints_dir), 'w')
        start_time = time.time()

        self.embedding = tfv2.cast(np.load('./data/weibo_vector.npy'), tf.float32)

        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        gpus = tfv2.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tfv2.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        train_dataset, dev_dataset = self.datamanager.get_train_dev_data()

        task_ner = []
        task_cws = []
        for i in range(self.setting.batch_size):
            task_ner.append([1, 0])
            task_cws.append([0, 1])

        # saver = tf.train.Saver(max_to_keep=3)
        best_f1_val = 0.0
        best_step = -1
        best_at_epoch = -1


        # each epoch
        for one_epoch in range(self.setting.num_epoches):
            info = "------epoch {}\n".format(one_epoch)
            print(info)
            out.write(info)

            for step, batch in train_dataset.shuffle(len(train_dataset)).batch(self.setting.batch_size).enumerate():
                X_train_batch, y_train_batch, att_mask_batch = batch
                # todo （batch_size,max_len,不知道是什么??）
                embedding_inputs = self.pretrained_model(X_train_batch, attention_mask=att_mask_batch)[0]
                # 普通嵌入
                # embedding_inputs = tfv2.nn.embedding_lookup(self.embedding, X_train_batch)

                # 计算没有加入pad之前的句子的长度，因为padding的id是0。参数 axis=1 表示在每个序列（行）上统计非零元素的数量。
                # 后续的模型训练需要inputs_length
                inputs_length_bacth = tfv2.math.count_nonzero(X_train_batch, 1)

                with tfv2.GradientTape() as tape:
                    # 前向传播，计算损失
                    logits, transition_params,loss = self.ner_model.single_task(embedding_inputs, inputs_length_bacth, y_train_batch,is_train=True)
                    current_step = int(self.global_step.numpy())
                    if current_step % 20 == 0:
                        temp = "step {},loss {}\n".format(current_step, loss)
                        print(temp)
                        out.write(temp)

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
                self.global_step.assign_add(1)

            # todo 测试
            # continue
            # # 每个epoch后验证模型
            val_f1_avg, val_res_str = self.validate(dev_dataset)
            time_span = (time.time() - start_time) / 60
            info = 'time consumption:%.2f(min), %s\n' % (time_span, val_res_str)
            print(info)
            out.write(info)

            if np.array(val_f1_avg).mean() > best_f1_val:
                best_f1_val = np.array(val_f1_avg).mean()
                best_at_epoch = one_epoch + 1
                self.checkpoint_manager.save()
                info = ('===== saved the new best model with f1: %.3f \n' % best_f1_val)
                print(info)
                out.write(info)


            # f1_val = validate(sess, test_word, test_label, test_length, setting, model)
            # if f1_val > best_f1_val:
            #     best_f1_val = f1_val
            #     best_step = current_step
            #     info = "=====update best F1, epoch = {},step = {} ,best_f1_score = {},current f1_val > best_f1_score".format(
            #         one_epoch, step, best_f1_val)
            #     print(info)
            #     out.write(info + "\n")
            #     saver.save(sess, save_path=save_path, global_step=current_step)

        info = '========= final best f1 = {},on epoch {},on step {}'.format(best_f1_val, best_at_epoch, best_step)
        print(info)
        out.write(info + "\n")

        extime = time.time() - start_time
        info = 'execute time is {} s\n,'.format(str(int(extime)))
        print(info)
        out.write(info)
        out.close()

    # 验证时需要加载对应的embedding
    def validate(self, val_dataset):
        # validation
        num_val_iterations = int(math.ceil(1.0 * len(val_dataset) / self.setting.batch_size))
        print('start evaluate engines...')
        loss_values = []
        val_results = {}
        val_labels_results = {}
        for label in self.cfg.suffix:
            val_labels_results.setdefault(label, {})
        measuring_metrics = ["precision", "recall", "f1", "accuracy"]
        for measure in measuring_metrics:
            val_results[measure] = 0
        for label, content in val_labels_results.items():
            for measure in measuring_metrics:
                if measure != 'accuracy':
                    val_labels_results[label][measure] = 0

        for val_batch in tqdm(val_dataset.batch(self.setting.batch_size)):
            X_val_batch, y_val_batch, att_mask_batch = val_batch

            embedding_inputs = self.pretrained_model(X_val_batch, attention_mask=att_mask_batch)[0]
            # 普通嵌入
            # embedding_inputs = tfv2.nn.embedding_lookup(self.embedding, X_val_batch)

            inputs_length_val = tf.math.count_nonzero(X_val_batch, 1)
            logits_val, transition_params_val,val_loss = (
                self.ner_model.single_task(embedding_inputs, inputs_length_val, y_val_batch,is_train=False)
            )
            batch_pred_sequence_val, _ = tfa.text.crf_decode(logits_val, transition_params_val, inputs_length_val)
            measures, lab_measures = MetricUtil.metrics(
                X_val_batch, y_val_batch, batch_pred_sequence_val, self.datamanager)

            for k, v in measures.items():
                val_results[k] += v
            for lab in lab_measures:
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
            print(info)
        return val_f1_avg, val_res_str

# def validate(sess,test_word,test_label,test_length,setting,model):
#     results = []
#     for j in range(len(test_word) // setting.batch_size):
#         word_batch = test_word[j * setting.batch_size:(j + 1) * setting.batch_size]
#         length_batch = test_length[j * setting.batch_size:(j + 1) * setting.batch_size]
#         label_batch = test_label[j * setting.batch_size:(j + 1) * setting.batch_size]
#         feed_dict = {}
#         feed_dict[model.input] = word_batch
#         feed_dict[model.sent_len] = length_batch
#         feed_dict[model.is_ner] = 1
#         logits, trans_params = sess.run([model.ner_project_logits, model.ner_trans_params], feed_dict)
#         # viterbi_sequences=decode(logits,trans_params,length_batch)
#         viterbi_sequences, viterbi_scores = tfa.text.crf_decode(logits, trans_params, length_batch)
#
#         result_batch = CommonUtil.evaluate(viterbi_sequences, label_batch, length_batch, word_batch)
#         results.append(result_batch)
#     f1 = CommonUtil.compute_f1(results)
#     return f1

if __name__ == "__main__":
    TrainSingle().train()