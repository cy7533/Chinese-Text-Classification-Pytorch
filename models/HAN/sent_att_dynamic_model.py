import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HAN.utils import matrix_mul, element_wise_mul, context_matrix_mul

"""
HAN 句子维度 动态context vector
"""


class SentAttDynamicNet(nn.Module):
    def __init__(self, sent_hidden_size, word_hidden_size, document_sent_hidden_size):
        """
        初始化模型
        :param sent_hidden_size: 当前层编码句子向量的隐向量维度
        :param word_hidden_size: 当前层编码词向量的隐向量维度
        :param document_sent_hidden_size: 外部传入的动态文档编码后的隐向量维度（用HAN编码的，是句子维度的长度）
        """
        super(SentAttDynamicNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.zeros(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.zeros(1, 2 * sent_hidden_size))
        self.context_weight_weight = nn.Parameter(torch.zeros(2 * document_sent_hidden_size, 2 * sent_hidden_size))
        self.context_weight_bias = nn.Parameter(torch.zeros(1, 2 * sent_hidden_size))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.sent_bias.data.normal_(mean, std)
        self.context_weight_weight.data.normal_(mean, std)
        self.context_weight_bias.data.normal_(mean, std)

    def forward(self, x, hidden_state, document_input):
        """
        param x: [seq_len, batch_size, 2 * word_hidden_size]
        param hidden_state: [1 * 2, batch_size, sent_hidden_size]
        param document_input: [batch_size, 2 * document_sent_hidden_size]
        """
        # f_output: [seq_len, batch_size, 2 * sent_hidden_size]
        # h_output: [1 * 2, batch_size, sent_hidden_size]
        f_output, h_output = self.gru(x, hidden_state)

        # context vector
        # context_weight: [batch_size, 2 * sent_hidden_size]
        context_weight = torch.mm(document_input, self.context_weight_weight)
        context_weight = context_weight + self.context_weight_bias.expand(context_weight.size()[0],
                                                                          self.context_weight_bias.size()[1])
        # context_weight: [2 * sent_hidden_size, batch_size]
        context_weight = context_weight.permute(1, 0)

        # output: [seq_len, batch_size, 2 * sent_hidden_size]
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        # output: [seq_len, batch_size] => [batch_size, seq_len]
        output = context_matrix_mul(output, context_weight).permute(1, 0)
        output = F.softmax(output, dim=1)
        # output: [batch_size, 2 * sent_hidden_size]
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        # output: [batch_size, num_classes]
        # output = self.fc(output)

        return output, h_output


if __name__ == "__main__":
    abc = SentAttDynamicNet(sent_hidden_size=128, word_hidden_size=128, document_sent_hidden_size=128)
    print(abc)