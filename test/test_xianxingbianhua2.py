import tensorflow as tf


class SharedBiLSTMWithAttention(tf.keras.Model):
    def __init__(self, lstm_units, attention_heads, output_units):
        super(SharedBiLSTMWithAttention, self).__init__()
        # 共享 BiLSTM 层
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=lstm_units, return_sequences=True))

        # 共享多头自注意力机制
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=attention_heads, key_dim=lstm_units * 2)

        # 线性变换层
        self.fc = tf.keras.layers.Dense(output_units, activation=None)

    def call(self, inputs):
        # BiLSTM 编码 (批大小, 序列长度, 维度)
        bilstm_output = self.bilstm(inputs)

        # 自注意力机制
        attention_output = self.multihead_attention(
            query=bilstm_output,
            value=bilstm_output,
            key=bilstm_output
        )

        # 全局平均池化，编码为单个向量
        pooled_output = tf.reduce_mean(attention_output, axis=1)

        # 线性变换投影到标量
        r = self.fc(pooled_output)

        return r


# 测试模型
if __name__ == "__main__":
    # 定义模型参数
    lstm_units = 64  # BiLSTM 的隐藏单元数
    attention_heads = 4  # 多头注意力机制的头数
    output_units = 1  # 线性变换的输出维度 (标量)

    # 初始化模型
    model = SharedBiLSTMWithAttention(lstm_units, attention_heads, output_units)

    # 创建输入数据 (批大小 32，序列长度 50，输入维度 128)
    x = tf.random.normal([32, 50, 128])

    # 前向传播
    output = model(x)

    print("模型输出形状:", output.shape)  # 输出形状应为 (32, 1)