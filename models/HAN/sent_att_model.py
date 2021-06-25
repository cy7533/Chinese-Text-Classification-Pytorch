import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HAN.utils import matrix_mul, element_wise_mul

"""
HAN 句子维度
"""


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size, word_hidden_size):
        """
        初始化模型
        :param sent_hidden_size: 当前层编码句子向量的隐向量维度
        :param word_hidden_size: 当前层编码词向量的隐向量维度
        """
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.zeros(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.zeros(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.zeros(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.sent_bias.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, x, hidden_state):
        """
        param x: [seq_len, batch, 2 * word_hidden_size]
        param hidden_state: [1 * 2, batch, sent_hidden_size]
        """
        # f_output: [seq_len, batch, 2 * sent_hidden_size]
        # h_output: [1 * 2, batch, sent_hidden_size]
        f_output, h_output = self.gru(x, hidden_state)
        # context vector
        # output: [seq_len, batch, 2 * sent_hidden_size]
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        # output: [seq_len, batch] => [batch, seq_len]
        output = matrix_mul(output, self.context_weight).squeeze(2).permute(1, 0)
        output = F.softmax(output, dim=1)
        # output: [batch, 2 * sent_hidden_size]
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        # output: [batch, num_classes]
        # output = self.fc(output)
        # print(output)

        return output, h_output


if __name__ == "__main__":
    abc = SentAttNet(sent_hidden_size=128, word_hidden_size=128)
    print(abc)
