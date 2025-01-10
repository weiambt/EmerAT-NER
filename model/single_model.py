
import base_model
import tensorflow as tfv2
import tensorflow.compat.v1 as tf

# 加这个和transfoemers不兼容
# tf.disable_eager_execution()
# tf.disable_v2_behavior()
import tensorflow_addons as tfa



class Model(tfv2.keras.Model):
    def __init__(self, setting,datamanager_src,datamanager_tgt=None):
        super(Model, self).__init__()
        self.lr = setting.lr
        self.word_dim = setting.word_dim
        self.lstm_dim = setting.lstm_dim
        self.num_units = setting.num_units
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
        # self.adv = adv

        self.tags_num_src = datamanager_src.max_label_number
        # self.tags_num_tgt = datamanager_tgt.max_label_number

        # todo 这个要传进来，在多模型中
        # self.task_label =




        self.transition_params = tfv2.Variable(tfv2.random.uniform(shape=(self.tags_num_src,self.tags_num_src)))

        self.shared_bilstm = tfv2.keras.layers.Bidirectional(
            tfv2.keras.layers.LSTM(self.lstm_dim, return_sequences=True, dropout=1 - self.in_keep_prob)
        )
        self.private_bilstm = tfv2.keras.layers.Bidirectional(
            tfv2.keras.layers.LSTM(self.lstm_dim, return_sequences=True, dropout=1 - self.in_keep_prob)
        )

        # 初始化 Dense 层，只需创建一次
        self.dense_q = tfv2.keras.layers.Dense(self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform())
        self.dense_k = tfv2.keras.layers.Dense(self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform())
        self.dense_v = tfv2.keras.layers.Dense(self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform())

        # 归一化
        self.beta = None
        self.gamma = None

        self.W_ner = tfv2.keras.layers.Dense(self.lstm_dim, activation="tanh", name="W_ner")

        self.dropout = tfv2.keras.layers.Dropout(rate=1 - self.keep_prob_src)

        self.logits_W_ner = tfv2.keras.layers.Dense(self.tags_num_src, name="logits_W_ner")

        # self.input = tf.placeholder(tf.int32, [None, self.num_steps])
        # # NER任务的label数组
        # self.label = tf.placeholder(tf.int32, [None, self.num_steps])
        # # CWS任务的label_数组，CWS任务调用这个，没啥用，可以直接用一个label
        # self.label_=tf.placeholder(tf.int32, [None,self.num_steps])
        # self.task_label = tf.placeholder(tf.int32, [None,2])
        # self.sent_len = tf.placeholder(tf.int32, [None])
        # self.is_ner = tf.placeholder(dtype=tf.int32)

        # with tf.variable_scope('word_embedding'):
        #     self.embedding = tf.get_variable(name='embedding', dtype=tf.float32,initializer=tf.cast(self.word_embed, tf.float32))

    def adversarial_loss(self,feature):
        flip_gradient = base_model.FlipGradientBuilder()
        feature=flip_gradient(feature)
        if self.is_train:
            feature=tf.nn.dropout(feature,self.keep_prob_src)
        W_adv = tf.get_variable(name='W_adv', shape=[2 * self.lstm_dim, self.task_num],
                                       dtype=tf.float32,
                                       initializer=tfv2.keras.initializers.GlorotUniform())
        b_adv = tf.get_variable(name='b_adv', shape=[self.task_num], dtype=tf.float32,
                                       initializer=tfv2.keras.initializers.GlorotUniform())
        logits = tf.nn.xw_plus_b(feature,W_adv,b_adv)
        adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.task_label))
        return adv_loss

    # def normalize(self,inputs, epsilon = 1e-8,scope="ln",reuse=None):
    #     with tf.variable_scope(scope, reuse=reuse):
    #         inputs_shape = inputs.get_shape()
    #         params_shape = inputs_shape[-1:]
    #         mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    #         beta = tf.Variable(tf.zeros(params_shape),dtype=tf.float32)
    #         gamma = tf.Variable(tf.ones(params_shape),dtype=tf.float32)
    #         normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
    #         outputs = gamma * normalized + beta
    #     return outputs
    def normalize(self, inputs):
        # 归一化参数：定义可训练参数 beta 和 gamma
        self.epsilon = 1e-8
        inputs_shape = tfv2.shape(inputs)  # 动态获取输入形状
        params_shape = inputs_shape[-1:]  # 取最后一个维度的形状
        if self.beta is None or self.gamma is None:
            self.beta = tfv2.Variable(tfv2.zeros(params_shape), dtype=tfv2.float32, trainable=True)
            self.gamma = tfv2.Variable(tfv2.ones(params_shape), dtype=tfv2.float32, trainable=True)

        mean, variance = tfv2.nn.moments(inputs, axes=[-1], keepdims=True)
        normalized = (inputs - mean) / tfv2.sqrt(variance + self.epsilon)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def self_attention(self,keys, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tfv2.nn.relu(tfv2.keras.layers.Dense(keys, self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform()))
            K = tfv2.nn.relu(tfv2.keras.layers.Dense(keys, self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform()))
            V = tfv2.nn.relu(tfv2.keras.layers.Dense(keys, self.num_units, kernel_initializer=tfv2.keras.initializers.GlorotUniform()))
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
                outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob_src)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
            outputs += keys
            outputs = self.normalize(outputs)
        return outputs

    def self_attention2(self, keys, scope='multihead_attention'):
        # Q, K, V 和注意力层的计算
        multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.num_units, dropout=1 - self.keep_prob_src)

        # 计算多头自注意力输出
        outputs = multi_head_attention(query=keys, value=keys, key=keys)

        # 计算 key_masks 和 query_masks
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # [batch_size, seq_len]
        query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # [batch_size, seq_len]

        # 转换为适用于 MultiHeadAttention 的形状
        key_masks = tf.tile(key_masks, [self.num_heads, 1])  # [num_heads * batch_size, seq_len]
        key_masks = tf.tile(tf.expand_dims(key_masks, 1),
                            [1, tf.shape(keys)[1], 1])  # [num_heads * batch_size, seq_len, seq_len]

        query_masks = tf.tile(query_masks, [self.num_heads, 1])  # [num_heads * batch_size, seq_len]
        query_masks = tf.tile(tf.expand_dims(query_masks, -1),
                              [1, 1, tf.shape(keys)[1]])  # [num_heads * batch_size, seq_len, seq_len]

        # **修复维度不匹配问题**
        query_masks = tf.expand_dims(query_masks, -1)  # 将 query_masks 转换为 [num_heads * batch_size, seq_len, 1]
        outputs *= tf.cast(query_masks,
                           dtype=tf.float32)  # [num_heads * batch_size, seq_len, num_units] * [num_heads * batch_size, seq_len, 1]

        # 残差连接 + 归一化
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

    @tfv2.function
    def single_task(self,input,sent_len,label,is_train):
        self.is_train = is_train

        if self.is_train:
            input = tfv2.nn.dropout(input,self.keep_prob_src)

        with tf.variable_scope('common_bilstm'):
            shared_bilstm = self.shared_bilstm(input)
            shared_output = self.self_attention3(shared_bilstm)

        with tf.variable_scope('ner_private_bilstm'):
            ner_private_output = self.private_bilstm(shared_output)
            ner_private_output = self.self_attention3(ner_private_output)

        # NER combined + CRF
        output = tfv2.concat([ner_private_output,shared_output],axis=-1)
        output = tfv2.reshape(output,[-1, 4 * self.lstm_dim])

        hidden_output_ner = self.W_ner(output)
        if self.is_train:
            hidden_output_ner = self.dropout(hidden_output_ner)

        ner_logits = self.logits_W_ner(hidden_output_ner)
        self.ner_project_logits = tf.reshape(ner_logits, [-1, self.num_steps, self.tags_num_src])
        with tf.variable_scope('ner_crf'):
            ner_crf_loss, self.transition_params = tfa.text.crf_log_likelihood(self.ner_project_logits, label, sent_len,transition_params=self.transition_params)
            self.ner_loss = tf.reduce_mean(-ner_crf_loss)

        # todo 为了单个任务能执行成功，这里设置成0
        self.loss = self.ner_loss
        return self.ner_project_logits,self.transition_params,self.loss

