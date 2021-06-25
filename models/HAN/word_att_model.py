import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HAN.utils import matrix_mul, element_wise_mul

"""
HAN 词维度
"""


class WordAttNet(nn.Module):
    def __init__(self, dict_len, embed_size, word_hidden_size):
        """
        初始化 HAN 词维度
        :param dict_len: 词典总的词数
        :param embed_size: embedding后的向量维度
        :param word_hidden_size: 当前层编码的词向量的隐向量维度
        """
        super(WordAttNet, self).__init__()

        self.word_weight = nn.Parameter(torch.zeros(2 * word_hidden_size, 2 * word_hidden_size))
        self.word_bias = nn.Parameter(torch.zeros(1, 2 * word_hidden_size))
        self.context_weight = nn.Parameter(torch.zeros(2 * word_hidden_size, 1))

        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size)
        self.gru = nn.GRU(embed_size, word_hidden_size, bidirectional=True)

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.word_bias.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, x, hidden_state):
        """
        param x: [seq_len, batch]
        hidden_state: [1 * 2, batch, word_hidden_size]
        """
        # print('x:', torch.isnan(x).int().sum())
        # output: [seq_len, batch, embed_size]
        output = self.lookup(x)
        # print('self.lookup:', torch.isnan(output).int().sum())


        # f_output: [seq_len, batch, 2 * word_hidden_size]
        # h_output: [1 * 2, batch, word_hidden_size]
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        # print('self.gru:', torch.isnan(f_output).int().sum())

        # context vector
        # output: [seq_len, batch, 2 * word_hidden_size]
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        # print('matrix_mul1:', torch.isnan(output).int().sum())
        # output: [seq_len, batch] => [batch, seq_len]
        output = matrix_mul(output, self.context_weight).squeeze(2).permute(1, 0)
        # print('matrix_mul2:', torch.isnan(output).int().sum())
        output = F.softmax(output, dim=1)
        # print('F.softmax:', torch.isnan(output).int().sum())
        # output: [1, batch, 2 * word_hidden_size]
        output = element_wise_mul(f_output, output.permute(1, 0))

        return output, h_output


if __name__ == "__main__":
    abc = WordAttNet(dict_len=1000, embed_size=300, word_hidden_size=128)
    print(abc)
