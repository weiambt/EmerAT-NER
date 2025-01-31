import torch
import torch.nn as nn

# 将共享 ＢｉＬＳＴＭ 层的结果输入到共享多头自注意力机制中编
#  码为单个向量，再通过线性变换投影到标量ｒ。

class SharedBiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, output_dim):
        super(SharedBiLSTMWithAttention, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=2 * hidden_dim, num_heads=num_heads,
                                                         batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)


    def forward(self, x):
        # BiLSTM
        bilstm_out, _ = self.bilstm(x)  # (batch_size, seq_len, 2*hidden_dim)

        # Multi-Head Self-Attention
        attn_out, _ = self.multihead_attention(bilstm_out, bilstm_out, bilstm_out)

        # Pooling
        pooled_out = torch.mean(attn_out, dim=1)  # Mean pooling

        # Linear Projection
        r = self.fc(pooled_out)
        return r


# Example Usage
model = SharedBiLSTMWithAttention(input_dim=128, hidden_dim=64, num_heads=4, output_dim=1)
x = torch.randn(32, 50, 128)  # Batch size: 32, Sequence length: 50, Input dim: 128
output = model(x)
print(output.shape)