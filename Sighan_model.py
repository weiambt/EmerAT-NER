import tensorflow.compat.v1 as tf
import base_model

tf.disable_eager_execution()

class Setting(object):
    def __init__(self):
        self.lr=0.001
        self.word_dim=100
        self.lstm_dim=140
        self.num_units=280
        self.num_heads=8
        self.num_steps=80
        self.keep_prob=0.7
        self.keep_prob1=0.6
        self.in_keep_prob=0.7
        self.out_keep_prob=0.6
        self.batch_size=64
        self.clip=5
        self.num_epoches=260
        self.adv_weight=0.06
        self.task_num=2
        self.ner_tags_num=7
        self.cws_tags_num=4

class TransferModel(object):
    def __init__(self,setting,word_embed,adv,is_train):
        self.lr = setting.lr
        self.word_dim = setting.word_dim
        self.lstm_dim = setting.lstm_dim
        self.num_units = setting.num_units
        self.num_steps = setting.num_steps
        self.num_heads = setting.num_heads
        self.keep_prob = setting.keep_prob
        self.keep_prob1 = setting.keep_prob1
        self.in_keep_prob = setting.in_keep_prob
        self.out_keep_prob = setting.out_keep_prob
        self.batch_size = setting.batch_size
        self.word_embed = word_embed
        self.clip = setting.clip
        self.adv_weight = setting.adv_weight
        self.task_num = setting.task_num
        self.adv = adv
        self.is_train = is_train
        self.ner_tags_num = setting.ner_tags_num
        self.cws_tags_num = setting.cws_tags_num

        self.input = tf.placeholder(tf.int32, [None, self.num_steps])
        self.label = tf.placeholder(tf.int32, [None, self.num_steps])
        self.label_=tf.placeholder(tf.int32, [None,self.num_steps])
        self.task_label = tf.placeholder(tf.int32, [None,2])
        self.sent_len = tf.placeholder(tf.int32, [None])
        self.is_ner = tf.placeholder(dtype=tf.int32)

        with tf.variable_scope('word_embedding'):
            # 定义变量embedding,值是self.word_embed 转换后的值
            self.embedding = tf.get_variable(name='embedding', dtype=tf.float32,
                                             initializer=tf.cast(self.word_embed, tf.float32))

    def adversarial_loss(self,feature):
        # 1、梯度反转
        # 翻转 feature 的梯度
        flip_gradient = base_model.FlipGradientBuilder()
        feature=flip_gradient(feature)
        if self.is_train:
            # 应用 dropout 正则化。这是另一种防止过拟合的技术，通过随机丢弃一些神经元的输出来实现。
            feature=tf.nn.dropout(feature,self.keep_prob1)

        # 2、Softmax
        # 两个变量 W_adv 和 b_adv，它们分别是权重和偏置项
        W_adv = tf.get_variable(name='W_adv', shape=[2 * self.lstm_dim, self.task_num],
                                       dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
        b_adv = tf.get_variable(name='b_adv', shape=[self.task_num], dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
        # 计算 logits，这是一个线性变换
        logits = tf.nn.xw_plus_b(feature,W_adv,b_adv)
        # 计算softmax 交叉熵损失,最后返回平均值
        adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.task_label))
        return adv_loss

    def normalize(self,inputs, epsilon = 1e-8,scope="ln",reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape),dtype=tf.float32)
            gamma = tf.Variable(tf.ones(params_shape),dtype=tf.float32)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta

        return outputs

    def self_attention(self,keys, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            K = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            V = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
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

    def multi_task(self):
        # 从embedding中查找嵌入矩阵中对应的向量
        # self.embedding 是嵌入矩阵（例如，词嵌入矩阵），self.input 是输入的索引序列。tf.nn.embedding_lookup函数会根据索引将嵌入矩阵中对应的嵌入向量提取出来，形成一个新的张量。
        input = tf.nn.embedding_lookup(self.embedding, self.input)
        if self.is_train:
            # 用于防止过拟合的正则化技术。
            input=tf.nn.dropout(input,self.keep_prob)
        with tf.variable_scope('common_bilstm'):
            # 1. BILSTM 生成前向和后向，然后调用Dropout
            shared_cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            shared_cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            if self.is_train:
                shared_cell_fw = tf.nn.rnn_cell.DropoutWrapper(shared_cell_fw, input_keep_prob=self.in_keep_prob,
                                                               output_keep_prob=self.out_keep_prob)
                shared_cell_bw = tf.nn.rnn_cell.DropoutWrapper(shared_cell_bw, input_keep_prob=self.in_keep_prob,
                                                               output_keep_prob=self.out_keep_prob)
            # output_fw, output_bw：这两个变量分别接收前向和后向 RNN 的输出。每个输出都是一个形状为 (batch_size, max_time, lstm_dim) 的张量，其中 batch_size 是批量大小，max_time 是序列的最大长度，lstm_dim 是 LSTM 单元的维度
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                shared_cell_fw, shared_cell_bw, input, sequence_length=self.sent_len, dtype=tf.float32)
            # 连接输出的两个向量
            shared_output = tf.concat([output_fw, output_bw], axis=2)

            # 2. 注意力机制,LSTM的输出经过attention层
            shared_output=self.self_attention(shared_output)
            # 新增一个维度。在 shared_output 的最后一个维度上添加一个新的维度，即将原来的三维张量（batch_size, max_time, lstm_dim）扩展为四维张量（batch_size, max_time, lstm_dim, 1）。
            shared_output1 = tf.expand_dims(shared_output,axis=-1)


            # 3. maxpool最大池化操作。
            # tf.nn.max_pool 是 TensorFlow 提供的最大池化操作。其功能是从指定窗口中取出最大值，用于下采样操作。
            max_pool_output = tf.nn.max_pool(shared_output1, ksize=[1, self.num_steps, 1, 1],
                                           strides=[1, self.num_steps, 1, 1], padding='SAME')
            # reshape重塑张量形状
            # -1：表示自动推断该维度的大小以适配整体形状。2 * self.lstm_dim：表示目标张量的每个样本包含 2 * self.lstm_dim 的特征。
            max_pool_output = tf.reshape(max_pool_output, [-1, 2 * self.lstm_dim])

        with tf.variable_scope('cws_private_bilstm'):
            cws_private_cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            cws_private_cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            if self.is_train:
                cws_private_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cws_private_cell_fw, input_keep_prob=self.in_keep_prob,
                                                               output_keep_prob=self.out_keep_prob)
                cws_private_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cws_private_cell_bw, input_keep_prob=self.in_keep_prob,
                                                               output_keep_prob=self.out_keep_prob)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cws_private_cell_fw, cws_private_cell_bw, input, sequence_length=self.sent_len, dtype=tf.float32)
            cws_private_output = tf.concat([output_fw, output_bw], axis=-1)
            # attention
            cws_private_output=self.self_attention(cws_private_output)

        with tf.variable_scope('ner_private_bilstm'):
            ner_private_cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            ner_private_cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            if self.is_train:
                ner_private_cell_fw = tf.nn.rnn_cell.DropoutWrapper(ner_private_cell_fw, input_keep_prob=self.in_keep_prob,
                                                               output_keep_prob=self.out_keep_prob)
                ner_private_cell_bw = tf.nn.rnn_cell.DropoutWrapper(ner_private_cell_bw, input_keep_prob=self.in_keep_prob,
                                                               output_keep_prob=self.out_keep_prob)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                ner_private_cell_fw, ner_private_cell_bw, input, sequence_length=self.sent_len, dtype=tf.float32)
            ner_private_output = tf.concat([output_fw, output_bw], axis=-1)
            # attention
            ner_private_output=self.self_attention(ner_private_output)

        # -----------------target task-----------------
        # 1. 将LSTM层的【NER输出】和【共享LSTM层的输出】进行拼接
        output = tf.concat([ner_private_output,shared_output],axis=-1)
        # 将最后一维reshape成 4 * self.lstm_dim，因为BILSTM有两层，并且还包含共享LSTM的
        output = tf.reshape(output,[-1, 4 * self.lstm_dim])

        # 2. 全连接层
        # 将拼接后的高维特征（4 * lstm_dim）通过线性变换和激活函数，映射到更低维的特征空间（lstm_dim），提取更紧凑的特征表示。
        # 权重矩阵 W_ner 将输入维度从 4 * lstm_dim 映射到 lstm_dim。
        # 使用 Xavier 初始化器 (tf.contrib.layers.xavier_initializer) 以保证初始值的适度大小。
        W_ner = tf.get_variable(name='W_ner', shape=[4 * self.lstm_dim, self.lstm_dim], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        b_ner = tf.get_variable(name='b_ner', shape=[self.lstm_dim], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        # tf.nn.xw_plus_b：计算线性变换 x * W + b。
        # tanh 双曲正切激活函数
        hidden_output = tf.tanh(tf.nn.xw_plus_b(output, W_ner, b_ner))
        if self.is_train:
            hidden_output=tf.nn.dropout(hidden_output,self.keep_prob1)
        # 得到 hidden_output 最后一维是lstm_dim

        # 3.计算logits，可以简化成 tf.layers.dense
        # ner_tags_num是实体个数
        logits_W = tf.get_variable(name='logits_weight', shape=[self.lstm_dim, self.ner_tags_num], dtype=tf.float32)
        logits_b = tf.get_variable(name='logits_bias', shape=[self.ner_tags_num], dtype=tf.float32)

        pred = tf.nn.xw_plus_b(hidden_output, logits_W, logits_b)
        self.ner_project_logits = tf.reshape(pred, [-1, self.num_steps, self.ner_tags_num])
        # ner_project_logits 最后一维是 ner_tags_num

        # 4、CRF层解码
        # self.ner_trans_params 保存了 CRF 的转移矩阵。
        with tf.variable_scope('ner_crf'):
            log_likelihood, self.ner_trans_params = tf.contrib.crf.crf_log_likelihood(inputs=self.ner_project_logits,
                                                                              tag_indices=self.label,
                                                                              sequence_lengths=self.sent_len)
        self.ner_loss=tf.reduce_mean(-log_likelihood)


        # -----------------source task-----------------
        output = tf.concat([cws_private_output, shared_output],axis=-1)
        output = tf.reshape(output, [-1, 4 * self.lstm_dim])
        W_cws = tf.get_variable(name='W_cws', shape=[4 * self.lstm_dim, self.lstm_dim], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        b_cws = tf.get_variable(name='b_cws', shape=[self.lstm_dim], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        hidden_output = tf.tanh(tf.nn.xw_plus_b(output, W_cws, b_cws))
        if self.is_train:
            hidden_output = tf.nn.dropout(hidden_output, self.keep_prob1)
        logits_W_cws = tf.get_variable(name='cws_weight', shape=[self.lstm_dim, self.cws_tags_num], dtype=tf.float32)
        logits_b_cws = tf.get_variable(name='cws_bias', shape=[self.cws_tags_num], dtype=tf.float32)
        pred = tf.nn.xw_plus_b(hidden_output, logits_W_cws, logits_b_cws)
        self.cws_project_logits = tf.reshape(pred, [-1, self.num_steps, self.cws_tags_num])
        with tf.variable_scope('cws_crf'):
            log_likelihood, self.cws_trans_params = tf.contrib.crf.crf_log_likelihood(inputs=self.cws_project_logits,
                                                                              tag_indices=self.label_,
                                                                              sequence_lengths=self.sent_len)
        self.cws_loss = tf.reduce_mean(-log_likelihood)



        # -----------------LOSS-----------------
        # adv_loss
        # 根据共享lstm层最大池化的结果计算
        self.adv_loss = self.adversarial_loss(max_pool_output)
        # todo loss = ner_loss + cws_loss + adv_weight * adv_loss
        self.loss=tf.cast(self.is_ner,tf.float32)*self.ner_loss+tf.cast((1-self.is_ner),tf.float32)*self.cws_loss+self.adv_weight*self.adv_loss

