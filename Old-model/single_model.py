
import base_model
import tensorflow as tfv2
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.disable_v2_behavior()
import tensorflow_addons as tfa

class Setting(object):
    def __init__(self):
        self.lr=0.001
        self.word_dim=100
        self.lstm_dim=120
        self.num_units=240
        self.num_heads=8
        # self.num_steps 通常表示输入序列的固定最大长度，不足的补padding，多的截断
        self.num_steps=80
        self.keep_prob=0.7
        self.keep_prob1=0.7
        self.in_keep_prob=0.7
        self.out_keep_prob=0.6
        self.batch_size=20
        self.clip=5
        self.num_epoches=3
        self.adv_weight=0.06
        self.task_num=2
        self.ner_tags_num=9
        self.cws_tags_num=4

class Model(object):
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
        # NER任务的label数组
        self.label = tf.placeholder(tf.int32, [None, self.num_steps])
        # CWS任务的label_数组，CWS任务调用这个，没啥用，可以直接用一个label
        self.label_=tf.placeholder(tf.int32, [None,self.num_steps])
        self.task_label = tf.placeholder(tf.int32, [None,2])
        self.sent_len = tf.placeholder(tf.int32, [None])
        self.is_ner = tf.placeholder(dtype=tf.int32)

        with tf.variable_scope('word_embedding'):
            self.embedding = tf.get_variable(name='embedding', dtype=tf.float32,initializer=tf.cast(self.word_embed, tf.float32))

    def adversarial_loss(self,feature):
        flip_gradient = base_model.FlipGradientBuilder()
        feature=flip_gradient(feature)
        if self.is_train:
            feature=tf.nn.dropout(feature,self.keep_prob1)
        W_adv = tf.get_variable(name='W_adv', shape=[2 * self.lstm_dim, self.task_num],
                                       dtype=tf.float32,
                                       initializer=tfv2.keras.initializers.GlorotUniform())
        b_adv = tf.get_variable(name='b_adv', shape=[self.task_num], dtype=tf.float32,
                                       initializer=tfv2.keras.initializers.GlorotUniform())
        logits = tf.nn.xw_plus_b(feature,W_adv,b_adv)
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

    def single_task(self):
        input = tfv2.nn.embedding_lookup(self.embedding, self.input)
        if self.is_train:
            input=tfv2.nn.dropout(input,self.keep_prob)
        with tf.variable_scope('common_bilstm'):
            shared_bilstm = tf.keras.layers.Bidirectional(
                tfv2.keras.layers.LSTM(self.lstm_dim, return_sequences=True, dropout=1 - self.in_keep_prob)
            )(input)
            shared_output = self.self_attention(shared_bilstm)
            max_pool_output = tf.reduce_max(shared_output, axis=1)
            # shared_cell_fw = tfv2.keras.layers.LSTMCell(self.lstm_dim)
            # shared_cell_bw = tfv2.keras.layers.LSTMCell(self.lstm_dim)
            # if self.is_train:
            #     shared_cell_fw = tf.nn.rnn_cell.DropoutWrapper(shared_cell_fw, input_keep_prob=self.in_keep_prob,
            #                                                    output_keep_prob=self.out_keep_prob)
            #     shared_cell_bw = tf.nn.rnn_cell.DropoutWrapper(shared_cell_bw, input_keep_prob=self.in_keep_prob,
            #                                                    output_keep_prob=self.out_keep_prob)
            # (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            #     shared_cell_fw, shared_cell_bw, input, sequence_length=self.sent_len, dtype=tf.float32)
            # shared_output = tf.concat([output_fw, output_bw], axis=2)
            # shared_output=self.self_attention(shared_output)
            # shared_output1 = tf.expand_dims(shared_output,axis=-1)
            # max_pool_output = tf.nn.max_pool(shared_output1, ksize=[1, self.num_steps, 1, 1],
            #                                strides=[1, self.num_steps, 1, 1], padding='SAME')
            # max_pool_output = tf.reshape(max_pool_output, [-1, 2 * self.lstm_dim])

        with tf.variable_scope('ner_private_bilstm'):
            ner_private_output = tfv2.keras.layers.Bidirectional(
                tfv2.keras.layers.LSTM(self.lstm_dim, return_sequences=True, dropout=1 - self.in_keep_prob)
            )(input)
            ner_private_output = self.self_attention(ner_private_output)

            # ner_private_cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            # ner_private_cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            # if self.is_train:
            #     ner_private_cell_fw = tf.nn.rnn_cell.DropoutWrapper(ner_private_cell_fw, input_keep_prob=self.in_keep_prob,
            #                                                    output_keep_prob=self.out_keep_prob)
            #     ner_private_cell_bw = tf.nn.rnn_cell.DropoutWrapper(ner_private_cell_bw, input_keep_prob=self.in_keep_prob,
            #                                                    output_keep_prob=self.out_keep_prob)
            #
            # (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            #     ner_private_cell_fw, ner_private_cell_bw, input, sequence_length=self.sent_len, dtype=tf.float32)
            # ner_private_output = tf.concat([output_fw, output_bw], axis=-1)
            # ner_private_output=self.self_attention(ner_private_output)

        # NER combined + CRF
        output = tfv2.concat([ner_private_output,shared_output],axis=-1)
        output = tfv2.reshape(output,[-1, 4 * self.lstm_dim])
        W_ner = tfv2.keras.layers.Dense(self.lstm_dim, activation="tanh", name="W_ner")
        hidden_output_ner = W_ner(output)
        if self.is_train:
            hidden_output_ner = tfv2.keras.layers.Dropout(rate=1 - self.keep_prob1)(hidden_output_ner)
        logits_W_ner = tfv2.keras.layers.Dense(self.ner_tags_num, name="logits_W_ner")
        ner_logits = logits_W_ner(hidden_output_ner)
        self.ner_project_logits = tf.reshape(ner_logits, [-1, self.num_steps, self.ner_tags_num])
        with tf.variable_scope('ner_crf'):
            ner_crf_loss, self.ner_trans_params = tfa.text.crf_log_likelihood(self.ner_project_logits, self.label, self.sent_len)
            self.ner_loss = tf.reduce_mean(-ner_crf_loss)


        # output = tf.concat([ner_private_output,shared_output],axis=-1)
        # output = tf.reshape(output,[-1, 4 * self.lstm_dim])
        # W_ner = tf.get_variable(name='W_ner', shape=[4 * self.lstm_dim, self.lstm_dim], dtype=tf.float32,
        #                         initializer=tf.contrib.layers.xavier_initializer())
        # b_ner = tf.get_variable(name='b_ner', shape=[self.lstm_dim], dtype=tf.float32,
        #                         initializer=tf.contrib.layers.xavier_initializer())
        # hidden_output = tf.tanh(tf.nn.xw_plus_b(output, W_ner, b_ner))
        # if self.is_train:
        #     hidden_output=tf.nn.dropout(hidden_output,self.keep_prob1)
        # logits_W = tf.get_variable(name='logits_weight', shape=[self.lstm_dim, self.ner_tags_num], dtype=tf.float32)
        # logits_b = tf.get_variable(name='logits_bias', shape=[self.ner_tags_num], dtype=tf.float32)
        # pred = tf.nn.xw_plus_b(hidden_output, logits_W, logits_b)
        # self.ner_project_logits = tf.reshape(pred, [-1, self.num_steps, self.ner_tags_num])
        # with tf.variable_scope('ner_crf'):
        #     log_likelihood, self.ner_trans_params = tf.contrib.crf.crf_log_likelihood(inputs=self.ner_project_logits,
        #                                                                       tag_indices=self.label,
        #                                                                       sequence_lengths=self.sent_len)
        # self.ner_loss=tf.reduce_mean(-log_likelihood)

        # todo 为了单个任务能执行成功，这里设置成0
        # self.cws_loss = tf.constant(0.0)

        # self.adv_loss = self.adversarial_loss(max_pool_output)
        self.loss = self.ner_loss

