
import base_model
import tensorflow as tfv2
import tensorflow.compat.v1 as tf

# 加这个和transfoemers不兼容
# tf.disable_eager_execution()
# tf.disable_v2_behavior()
import tensorflow_addons as tfa

class TransferModel(tfv2.keras.Model):
    def __init__(self,setting,datamanager_src,datamanager_tgt):
        super(TransferModel, self).__init__()
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

        # todo 这里先不改
        self.dropout_src = tfv2.keras.layers.Dropout(rate=1 - self.keep_prob_src)
        self.dropout_tgt = tfv2.keras.layers.Dropout(rate=1 - self.keep_prob_tgt)


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

        if self.is_target is False:
            self.loss = self.adv_loss * self.adv_weight + self.ner_loss_src
        else:
            self.loss = self.adv_loss * self.adv_weight + self.ner_loss_tgt

        transition_params = self.transition_params_tgt if is_target else self.transition_params_src
        return self.ner_project_logits, transition_params, self.loss

        # with tf.variable_scope('cws_private_bilstm'):
        #     cws_private_output = tfv2.keras.layers.Bidirectional(
        #         tfv2.keras.layers.LSTM(self.lstm_dim, return_sequences=True, dropout=1 - self.in_keep_prob)
        #     )(input)
        #     cws_private_output = self.self_attention(cws_private_output)

        # with tf.variable_scope('ner_private_bilstm'):
            # ner_private_output = tfv2.keras.layers.Bidirectional(
            #     tfv2.keras.layers.LSTM(self.lstm_dim, return_sequences=True, dropout=1 - self.in_keep_prob)
            # )(input)
            # ner_private_output = self.self_attention(ner_private_output)

        # NER combined + CRF
        # output = tfv2.concat([ner_private_output,shared_output],axis=-1)
        # output = tfv2.reshape(output,[-1, 4 * self.lstm_dim])
        # W_ner = tfv2.keras.layers.Dense(self.lstm_dim, activation="tanh", name="W_ner")
        # hidden_output_ner = W_ner(output)
        # if self.is_train:
        #     hidden_output_ner = tfv2.keras.layers.Dropout(rate=1 - self.keep_prob1)(hidden_output_ner)
        # logits_W_ner = tfv2.keras.layers.Dense(self.ner_tags_num, name="logits_W_ner")
        # ner_logits = logits_W_ner(hidden_output_ner)
        # self.ner_project_logits = tf.reshape(ner_logits, [-1, self.num_steps, self.ner_tags_num])
        # with tf.variable_scope('ner_crf'):
        #     ner_crf_loss, self.ner_trans_params = tfa.text.crf_log_likelihood(self.ner_project_logits, self.label, self.sent_len)
        #     self.ner_loss = tf.reduce_mean(-ner_crf_loss)

        # CWS combined CRF
        # output = tf.concat([cws_private_output, shared_output],axis=-1)
        # output = tf.reshape(output, [-1, 4 * self.lstm_dim])
        # W_cws = tf.get_variable(name='W_cws', shape=[4 * self.lstm_dim, self.lstm_dim], dtype=tf.float32,
        #                         initializer=tf.contrib.layers.xavier_initializer())
        # b_cws = tf.get_variable(name='b_cws', shape=[self.lstm_dim], dtype=tf.float32,
        #                         initializer=tf.contrib.layers.xavier_initializer())
        # hidden_output = tf.tanh(tf.nn.xw_plus_b(output, W_cws, b_cws))
        # if self.is_train:
        #     hidden_output = tf.nn.dropout(hidden_output, self.keep_prob1)
        # logits_W_cws = tf.get_variable(name='cws_weight', shape=[self.lstm_dim, self.cws_tags_num], dtype=tf.float32)
        # logits_b_cws = tf.get_variable(name='cws_bias', shape=[self.cws_tags_num], dtype=tf.float32)
        # pred = tf.nn.xw_plus_b(hidden_output, logits_W_cws, logits_b_cws)
        # self.cws_project_logits = tf.reshape(pred, [-1, self.num_steps, self.cws_tags_num])
        # with tf.variable_scope('cws_crf'):
        #     log_likelihood, self.cws_trans_params = tf.contrib.crf.crf_log_likelihood(inputs=self.cws_project_logits,
        #                                                                       tag_indices=self.label_,
        #                                                                       sequence_lengths=self.sent_len)
        # self.cws_loss = tf.reduce_mean(-log_likelihood)


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

        self.loss = self.adv_loss * self.adv_weight + self.ner_loss_src + self.ner_loss_tgt

        return self.loss

        # transition_params = self.transition_params_tgt if is_target else self.transition_params_src
        # return self.ner_project_logits, transition_params, self.loss



    def adversarial_loss(self,feature):
        flip_gradient = base_model.FlipGradientBuilder()
        feature=flip_gradient(feature)
        if self.is_train:
            keep_prob = self.keep_prob_tgt if self.is_target else self.keep_prob_src
            feature=tf.nn.dropout(feature,keep_prob)
        # W_adv = tf.get_variable(name='W_adv', shape=[2 * self.lstm_dim, self.task_num],
        #                                dtype=tf.float32,
        #                                initializer=tfv2.keras.initializers.GlorotUniform())
        # b_adv = tf.get_variable(name='b_adv', shape=[self.task_num], dtype=tf.float32,
        #                                initializer=tfv2.keras.initializers.GlorotUniform())
        # logits = tf.nn.xw_plus_b(feature,W_adv,b_adv)

        logits = tfv2.matmul(feature, self.W_adv) + self.b_adv
        # minmiax将特征映射到[0,1]中：softmax分类到[0,1]中，用交叉熵计算损失
        adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.task_label))
        return adv_loss

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



