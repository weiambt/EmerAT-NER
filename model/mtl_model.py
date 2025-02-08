# from safetensors import paddle

import base_model
import tensorflow as tfv2
import tensorflow.compat.v1 as tf

# 加这个和transfoemers不兼容
# tf.disable_eager_execution()
# tf.disable_v2_behavior()
import tensorflow_addons as tfa

# 尝试解决服务器使用GPU报错
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from model.LossType import LossType

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class TransferModel(tfv2.keras.Model):
    def __init__(self,logger,setting,datamanager_src,datamanager_tgt):
        super(TransferModel, self).__init__()
        self.logger = logger
        self.lr = setting.lr
        self.word_dim = setting.word_dim
        self.lstm_dim = setting.lstm_dim
        self.num_units = setting.num_units
        # todo 换名字
        self.num_steps = setting.num_steps
        self.num_heads = setting.num_heads
        self.keep_prob_tgt = setting.keep_prob_tgt
        self.keep_prob_src = setting.keep_prob_src
        self.in_keep_prob = setting.in_keep_prob
        self.out_keep_prob = setting.out_keep_prob
        self.batch_size = setting.batch_size
        # self.word_embed = word_embed
        self.clip = setting.clip
        self.adv_weight = setting.adv_weight
        self.task_num = setting.task_num
        self.loss_type = setting.loss_type
        self.focal_alpha = setting.focal_alpha
        self.focal_gamma = setting.focal_gamma

        self.tags_num_src = datamanager_src.max_label_number
        self.tags_num_tgt = datamanager_tgt.max_label_number

        self.shared_bilstm = tfv2.keras.layers.Bidirectional(
            tfv2.keras.layers.LSTM(self.lstm_dim, return_sequences=True, dropout=1 - self.in_keep_prob)
        )
        self.private_bilstm_src = tfv2.keras.layers.Bidirectional(
            tfv2.keras.layers.LSTM(self.lstm_dim, return_sequences=True, dropout=1 - self.in_keep_prob)
        )
        self.private_bilstm_tgt = tfv2.keras.layers.Bidirectional(
            tfv2.keras.layers.LSTM(self.lstm_dim, return_sequences=True, dropout=1 - self.in_keep_prob)
        )

        self.transition_params_src = tfv2.Variable(tfv2.random.uniform(shape=(self.tags_num_src,self.tags_num_src),name="transition_params_src"))
        self.transition_params_tgt = tfv2.Variable(tfv2.random.uniform(shape=(self.tags_num_tgt,self.tags_num_tgt),name="transition_params_tgt"))

        # 自注意力 Dense 层
        self.dense_q = tfv2.keras.layers.Dense(self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform())
        self.dense_k = tfv2.keras.layers.Dense(self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform())
        self.dense_v = tfv2.keras.layers.Dense(self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform())

        # 归一化
        self.beta = None
        self.gamma = None

        # 对抗参数
        initializer = tfv2.keras.initializers.GlorotUniform()
        self.W_adv = tfv2.Variable(initializer(shape=[2 * self.lstm_dim, self.task_num]), name='W_adv', dtype=tf.float32)
        self.b_adv = tfv2.Variable(tf.zeros(shape=[self.task_num]), name='b_adv', dtype=tf.float32)

        self.W_src = tfv2.keras.layers.Dense(self.lstm_dim, activation="tanh", name="W_src")
        self.W_tgt = tfv2.keras.layers.Dense(self.lstm_dim, activation="tanh", name="W_tgt")

        self.logits_W_src = tfv2.keras.layers.Dense(self.tags_num_src, name="logits_W_src")
        self.logits_W_tgt = tfv2.keras.layers.Dense(self.tags_num_tgt, name="logits_W_tgt")

        # 线性变化成标量
        self.linear_layer =  tf.keras.layers.Dense(1, activation=None)

        # todo 这里先不改
        self.dropout_src = tfv2.keras.layers.Dropout(rate=1 - self.keep_prob_src)
        self.dropout_tgt = tfv2.keras.layers.Dropout(rate=1 - self.keep_prob_tgt)

        # 同方差，损失平衡
        self.log_var2s = tfv2.Variable(tf.zeros([self.task_num], dtype=tf.float32))
        # x = paddle.zeros([self.task_num], dtype='float32')
        # self.log_var2s = paddle.create_parameter(
        #     shape=x.shape,
        #     dtype=str(x.numpy().dtype),
        #     default_initializer=paddle.nn.initializer.Assign(x))


    @tfv2.function
    def multi_task(self,input,sent_len,label,is_target,is_train,task_label):
        print("Trainable variables:", self.trainable_variables)
        self.is_train = is_train
        self.is_target = is_target
        self.task_label = task_label

        if self.is_train:
            if self.is_target:
                input=tfv2.nn.dropout(input,self.keep_prob_tgt)
            else:
                input=tfv2.nn.dropout(input,self.keep_prob_src)

        with tfv2.name_scope('shared_bilstm'):
            shared_bilstm = self.shared_bilstm(input)
            shared_bilstm_output = self.self_attention3(shared_bilstm)
            # todo max pool最大池化操作，使用 tf.reduce_max 沿着 axis=1列方向 进行最大值池化
            max_pool_output = tfv2.reduce_max(shared_bilstm_output, axis=1)


        if self.is_target is False:
            # source lstm
            with tfv2.name_scope('source_bilstm'):
                src_private_output = self.private_bilstm_src(shared_bilstm_output)
                src_private_output = self.self_attention3(src_private_output)
            # common + private lstm + CRF
            output = tfv2.concat([src_private_output, shared_bilstm_output], axis=-1)
            output = tfv2.reshape(output, [-1, 4 * self.lstm_dim])
            hidden_output_ner = self.W_src(output)
            if self.is_train:
                hidden_output_ner = self.dropout_src(hidden_output_ner)

            ner_logits = self.logits_W_src(hidden_output_ner)
            self.ner_project_logits = tf.reshape(ner_logits, [-1, self.num_steps, self.tags_num_src])
            with tfv2.name_scope('ner_crf'):
                ner_crf_loss, self.transition_params_src = tfa.text.crf_log_likelihood(self.ner_project_logits,
                                                                                       label,
                                                                                        sent_len,
                                                                                        transition_params=self.transition_params_src)
                self.ner_loss_src = tf.reduce_mean(-ner_crf_loss)
        else:
            with tfv2.name_scope('target_bilstm'):
                tgt_private_output = self.private_bilstm_tgt(shared_bilstm_output)
                tgt_private_output = self.self_attention3(tgt_private_output)

            # common + private lstm + CRF
            output = tfv2.concat([tgt_private_output, shared_bilstm_output], axis=-1)
            output = tfv2.reshape(output, [-1, 4 * self.lstm_dim])
            hidden_output_ner = self.W_tgt(output)
            if self.is_train:
                hidden_output_ner = self.dropout_tgt(hidden_output_ner)

            ner_logits = self.logits_W_tgt(hidden_output_ner)
            self.ner_project_logits = tf.reshape(ner_logits, [-1, self.num_steps, self.tags_num_tgt])
            with tfv2.name_scope('ner_crf'):
                ner_crf_loss, self.transition_params_tgt = tfa.text.crf_log_likelihood(self.ner_project_logits, label,
                                                                                   sent_len,
                                                                                   transition_params=self.transition_params_tgt)
                self.ner_loss_tgt = tf.reduce_mean(-ner_crf_loss)

        # 对抗损失
        self.adv_loss = self.adversarial_loss(max_pool_output)
        # self.loss=tf.cast(self.is_ner,tf.float32)*self.ner_loss+tf.cast((1-self.is_ner),tf.float32)*self.cws_loss+self.adv_weight*self.adv_loss

        # 平衡CRF损失
        if self.loss_type == LossType.BALANCE_crfloss:
            if self.is_target is False:
                self.loss = self.balance_loss(0, self.ner_loss_src) + self.adv_loss
            else:
                self.loss = self.balance_loss(1, self.ner_loss_tgt) + self.adv_loss
        # 正常损失
        else:
            if self.is_target is False:
                self.loss = self.ner_loss_src + self.adv_loss
            else:
                self.loss = self.ner_loss_tgt + self.adv_loss
        self.logger.info('adv_loss = {}'.format(self.adv_loss))
        self.logger.info('final_loss = {}'.format(self.loss))
        transition_params = self.transition_params_tgt if is_target else self.transition_params_src
        return self.ner_project_logits, transition_params, self.loss


    @tfv2.function
    def multi_task_init(self,input,sent_len,label,is_train,task_label):
        """
            1. 为了初始化target的参数
            2、如果需要两个任务共同计算损失后，再传播，就用这个代码
        """

        self.is_train = is_train
        self.task_label = task_label
        self.is_target = True

        if self.is_train:
            input=tfv2.nn.dropout(input,self.keep_prob_tgt)

            # if self.is_target:
            #     input=tfv2.nn.dropout(input,self.keep_prob_tgt)
            # else:
            #     input=tfv2.nn.dropout(input,self.keep_prob_src)

        with tfv2.name_scope('shared_bilstm'):
            shared_bilstm = self.shared_bilstm(input)
            shared_bilstm_output = self.self_attention3(shared_bilstm)
            # todo max pool最大池化操作，使用 tf.reduce_max 沿着 axis=1列方向 进行最大值池化
            max_pool_output = tfv2.reduce_max(shared_bilstm_output, axis=1)

        # source lstm
        with tfv2.name_scope('source_bilstm'):
            # ToDo 这里为什么用的是共享lstm的输出
            src_private_output = self.private_bilstm_src(input)
            src_private_output = self.self_attention3(src_private_output)

        # common + private lstm + CRF
        output = tfv2.concat([src_private_output, shared_bilstm_output], axis=-1)
        output = tfv2.reshape(output, [-1, 4 * self.lstm_dim])
        hidden_output_ner = self.W_src(output)
        if self.is_train:
            hidden_output_ner = self.dropout_src(hidden_output_ner)

        ner_logits = self.logits_W_src(hidden_output_ner)
        self.ner_project_logits = tf.reshape(ner_logits, [-1, self.num_steps, self.tags_num_src])
        with tfv2.name_scope('ner_crf'):
            ner_crf_loss, self.transition_params_src = tfa.text.crf_log_likelihood(self.ner_project_logits,
                                                                                   label,
                                                                                    sent_len,
                                                                                    transition_params=self.transition_params_src)
            self.ner_loss_src = tf.reduce_mean(-ner_crf_loss)

        with tfv2.name_scope('target_bilstm'):
            tgt_private_output = self.private_bilstm_tgt(input_tgt)
            tgt_private_output = self.self_attention3(tgt_private_output)

        # common + private lstm + CRF
        output = tfv2.concat([tgt_private_output, shared_bilstm_output], axis=-1)
        output = tfv2.reshape(output, [-1, 4 * self.lstm_dim])
        hidden_output_ner = self.W_tgt(output)
        if self.is_train:
            hidden_output_ner = self.dropout_tgt(hidden_output_ner)

        ner_logits = self.logits_W_tgt(hidden_output_ner)
        self.ner_project_logits = tf.reshape(ner_logits, [-1, self.num_steps, self.tags_num_tgt])
        with tfv2.name_scope('ner_crf'):
            ner_crf_loss, self.transition_params_tgt = tfa.text.crf_log_likelihood(self.ner_project_logits,
                                                                                   label_tgt,
                                                                                sent_len_tgt,
                                                                               transition_params=self.transition_params_tgt)
            self.ner_loss_tgt = tf.reduce_mean(-ner_crf_loss)

        # 对抗损失
        self.adv_loss = self.adversarial_loss(max_pool_output)
        # self.loss=tf.cast(self.is_ner,tf.float32)*self.ner_loss+tf.cast((1-self.is_ner),tf.float32)*self.cws_loss+self.adv_weight*self.adv_loss

        self.loss = self.adv_loss + self.ner_loss_src + self.ner_loss_tgt

        return self.loss

        # transition_params = self.transition_params_tgt if is_target else self.transition_params_src
        # return self.ner_project_logits, transition_params, self.loss

    def normalize(self, inputs):
        # 归一化参数：定义可训练参数 beta 和 gamma
        self.epsilon = 1e-8
        inputs_shape = tfv2.shape(inputs)  # 动态获取输入形状
        params_shape = inputs_shape[-1:]  # 取最后一个维度的形状
        if self.beta is None or self.gamma is None:
            self.beta = tfv2.Variable(tfv2.zeros(params_shape), dtype=tfv2.float32, trainable=True,name="beta")
            self.gamma = tfv2.Variable(tfv2.ones(params_shape), dtype=tfv2.float32, trainable=True,name="gamma")

        mean, variance = tfv2.nn.moments(inputs, axes=[-1], keepdims=True)
        normalized = (inputs - mean) / tfv2.sqrt(variance + self.epsilon)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def self_attention(self,keys, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform()))
            K = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform()))
            V = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform()))
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [self.num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            query_masks = tf.tile(query_masks, [self.num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks
            if self.is_train:
                outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob1)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
            outputs += keys
            outputs = self.normalize(outputs)
        return outputs

    def self_attention3(self, keys, scope='multihead_attention'):
        # 使用 Keras 层替代原始的 tfv2.keras.layers.Dense
        Q = tfv2.nn.relu(self.dense_q(keys))
        K = tfv2.nn.relu(self.dense_k(keys))
        V = tfv2.nn.relu(self.dense_v(keys))

        # 拆分并拼接（多头注意力部分）
        Q_ = tfv2.concat(tfv2.split(Q, self.num_heads, axis=2), axis=0)
        K_ = tfv2.concat(tfv2.split(K, self.num_heads, axis=2), axis=0)
        V_ = tfv2.concat(tfv2.split(V, self.num_heads, axis=2), axis=0)

        # 计算注意力得分
        outputs = tfv2.matmul(Q_, tfv2.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.shape.as_list()[-1] ** 0.5)  # 计算缩放因子

        # 计算 mask
        key_masks = tfv2.sign(tfv2.abs(tfv2.reduce_sum(keys, axis=-1)))
        key_masks = tfv2.tile(key_masks, [self.num_heads, 1])
        key_masks = tfv2.tile(tfv2.expand_dims(key_masks, 1), [1, tfv2.shape(keys)[1], 1])

        # 应用 padding
        paddings = tfv2.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tfv2.where(tfv2.equal(key_masks, 0), paddings, outputs)

        # 归一化和 softmax
        outputs = tfv2.nn.softmax(outputs)

        # 计算 query_masks
        query_masks = tfv2.sign(tfv2.abs(tfv2.reduce_sum(keys, axis=-1)))
        query_masks = tfv2.tile(query_masks, [self.num_heads, 1])
        query_masks = tfv2.tile(tfv2.expand_dims(query_masks, -1), [1, 1, tfv2.shape(keys)[1]])

        # 应用 query_masks
        outputs *= query_masks

        # Dropout
        if self.is_train:
            outputs = tfv2.nn.dropout(outputs, self.keep_prob_src)

        # 计算输出
        outputs = tfv2.matmul(outputs, V_)

        # 拼接头
        outputs = tfv2.concat(tfv2.split(outputs, self.num_heads, axis=0), axis=2)

        # 残差连接
        outputs += keys

        # 归一化
        outputs = self.normalize(outputs)

        return outputs

    # 损失函数计算
    def adversarial_loss(self,feature):
        flip_gradient = base_model.FlipGradientBuilder()
        feature=flip_gradient(feature)
        if self.is_train:
            keep_prob = self.keep_prob_tgt if self.is_target else self.keep_prob_src
            feature=tf.nn.dropout(feature,keep_prob)

        logits = tfv2.matmul(feature, self.W_adv) + self.b_adv

        adv_loss = -1
        if self.loss_type == LossType.AT:
            # 2018 AT模型的损失函数
            # minmiax将特征映射到[0,1]中：softmax分类到[0,1]中，用交叉熵计算损失
            adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.task_label))
            adv_loss = adv_loss * self.adv_weight
        elif self.loss_type == LossType.FOCAL:
            adv_loss = tf.reduce_mean(self.focal_loss(logits, self.task_label, alpha=self.focal_alpha, gamma=self.focal_gamma))
            # todo 这里不确定要不要加权重,感觉可以加，pang这里就是加了权重0.06
            adv_loss = adv_loss * self.adv_weight
        elif self.loss_type == LossType.BALANCE_advloss:
            adv_loss_at = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.task_label))
            adv_loss_focal = tf.reduce_mean(self.focal_loss(logits, self.task_label, alpha=self.focal_alpha, gamma=self.focal_gamma))
            adv_loss = self.balance_loss(0, adv_loss_at) + self.balance_loss(1, adv_loss_focal)
            # todo 如果结果不好尝试在这里加权重
            # adv_loss = adv_loss * self.adv_weight
        elif self.loss_type == LossType.BALANCE_crfloss:
            # todo 如果平衡CRF损失，对抗损失用 focal loss，这里确定下用不用权重，目前有权重
            adv_loss = tf.reduce_mean(self.focal_loss(logits, self.task_label, alpha=self.focal_alpha, gamma=self.focal_gamma))
            adv_loss = adv_loss * self.adv_weight
        return adv_loss

    def focal_loss(self,logits, labels, weights=None, alpha=0.25, gamma=2):
        # 将 logits 转换为概率分布
        logits = tfv2.nn.softmax(logits, axis=1)

        # 将 labels 转换为 float32 类型
        labels = tfv2.cast(labels, dtype=tfv2.float32)

        # 如果 labels 的维度小于 logits，则进行 one-hot 编码
        if labels.shape.ndims < logits.shape.ndims:
            labels = tfv2.one_hot(tfv2.cast(labels, dtype=tfv2.int32), depth=tfv2.shape(logits)[-1], axis=-1)

        # 创建与 logits 相同形状的全零张量
        zeros = tfv2.zeros_like(logits, dtype=logits.dtype)

        # 计算正类和负类的 logits 概率
        pos_logits_prob = tfv2.where(labels > zeros, labels - logits, zeros)
        neg_logits_prob = tfv2.where(labels > zeros, zeros, logits)

        # 计算 focal loss
        cross_entropy = - alpha * (pos_logits_prob ** gamma) * tfv2.math.log(tfv2.clip_by_value(logits, 1e-8, 1.0)) \
                        - (1 - alpha) * (neg_logits_prob ** gamma) * tfv2.math.log(tfv2.clip_by_value(1.0 - logits, 1e-8, 1.0))

        # 如果有 weights，则加权
        if weights is not None:
            if weights.shape.ndims < logits.shape.ndims:
                weights = tfv2.expand_dims(weights, axis=-1)
            cross_entropy = cross_entropy * weights

        # 返回最终的 cross entropy
        return cross_entropy

    # def mtl_loss(self, task_index, task_loss):
    #     loss = 0
    #     # 获取当前任务的对数方差
    #     log_var2 = self.log_var2s[task_index]
    #
    #     # 计算当前任务的损失
    #     pre = paddle.exp(-log_var2)  # 动态权重
    #     loss = paddle.sum(pre * task_loss + log_var2, axis=-1)  # 动态平衡损失
    #     return paddle.mean(loss)

    def balance_loss(self, task_index, task_loss):
        print("损失平衡----")
        # 获取当前任务的对数方差
        log_var2 = self.log_var2s[task_index]

        # 计算动态权重
        pre_weight = tfv2.exp(-log_var2)

        # 计算加权损失和正则化项
        weighted_loss = pre_weight * task_loss
        regularization = log_var2

        # 对加权损失和正则化项求和，并指定在最后一个维度上进行
        total_loss = tfv2.reduce_sum(weighted_loss + regularization)

        # 计算所有样本损失的平均值
        mean_loss = tfv2.reduce_mean(total_loss)

        return mean_loss

    def focal_loss2(self,logits, labels, weights=None, alpha=0.25, gamma=2):
        # 经过线性变化，得到标量
        pt = self.linear_layer(logits)

        # Step 1: 对 logits 进行 softmax 激活，转换为概率分布
        logits = tf.nn.softmax(logits, axis=-1)

        # Step 2: 将 labels 转换为浮点型张量，确保后续计算的一致性
        labels = tf.cast(labels, dtype=tf.float32)

        # Step 3: 如果 labels 的维度比 logits 的维度少（未进行 one-hot 编码），进行 one-hot 编码
        # 假设 logits 的最后一维是类别数 depth
        if labels.shape.ndims < logits.shape.ndims:
            labels = tf.one_hot(labels, depth=tf.shape(logits)[-1])

        # Step 4: 计算正样本和负样本的损失权重
        # pos_logits_prob 是正样本的权重，用于计算正样本部分的 Focal Loss
        # neg_logits_prob 是负样本的权重，用于计算负样本部分的 Focal Loss
        pos_logits_prob = labels * (1 - logits)  # (1 - p_t) 对应正样本
        neg_logits_prob = (1 - labels) * logits  # p_t 对应负样本

        # Step 5: 计算正负样本的 Focal Loss
        # 使用 gamma 对样本权重进行调整，使得低置信度样本的影响更大
        # 使用 tf.math.log 和 tf.clip_by_value 来确保数值稳定性，避免 log(0) 的情况
        cross_entropy = -alpha * (pos_logits_prob ** gamma) * tf.math.log(tf.clip_by_value(logits, 1e-8, 1.0)) \
                        - (1 - alpha) * (neg_logits_prob ** gamma) * tf.math.log(
            tf.clip_by_value(1 - logits, 1e-8, 1.0))

        # Step 6: 如果提供了 weights，则对损失进行加权
        if weights is not None:
            # 如果 weights 的维度小于 logits 的维度，将其扩展为与 logits 的维度一致
            if weights.shape.ndims < logits.shape.ndims:
                weights = tf.expand_dims(weights, axis=-1)
            # 将权重应用到损失上
            cross_entropy = cross_entropy * weights

        # Step 7: 计算最终损失
        # 先对类别维度（最后一维）求和，再对样本维度求平均，返回一个标量损失值
        return tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=-1))
