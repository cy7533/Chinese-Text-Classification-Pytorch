import torch
import torch.nn as nn

from models.HAN.word_att_dynamic_model import WordAttDynamicNet
from models.HAN.sent_att_dynamic_model import SentAttDynamicNet

"""
HAN with dynamic document context
"""


class HierAttDynamicNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, dict_len, embed_size,
                 document_sent_hidden_size, device):
        """
        :param word_hidden_size: 当前层编码的词向量的隐向量维度
        :param sent_hidden_size: 当前层编码的句子向量的隐向量维度
        :param batch_size: 批处理大小
        :param dict_len: 词典总的词数
        :param embed_size: embedding后的向量维度
        :param document_sent_hidden_size: 外部传入的动态文档编码后的隐向量维度（用HAN编码的，是句子维度的长度）
        :param device: 训练gpu(cuda)或cpu
        """
        super(HierAttDynamicNet, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size

        self.document_sent_hidden_size = document_sent_hidden_size
        self.word_att_net = WordAttDynamicNet(dict_len, embed_size, word_hidden_size, document_sent_hidden_size)
        self.sent_att_net = SentAttDynamicNet(sent_hidden_size, word_hidden_size, document_sent_hidden_size)

        self.word_hidden_state = None
        self.sent_hidden_state = None
        self.init_hidden_state()

    def init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.randn(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.randn(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.to(self.device)
            self.sent_hidden_state = self.sent_hidden_state.to(self.device)

    def forward(self, x, document_input):
        """
        param x: [batch, max_sent_length, max_word_length]
        param document_input: [batch_size, 2 * document_sent_hidden_size]
        """
        output_list = []
        # input: [max_sent_length, batch, max_word_length]
        x = x.permute(1, 0, 2)
        # i: [batch, max_word_length] => [max_word_length, batch]
        for i in x:
            # output: [batch, 2 * word_hidden_size]
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state, document_input)
            # output_list = max_sent_length * [batch, 2 * word_hidden_size]
            output_list.append(output)
        # output: [max_sent_length, batch, 2 * word_hidden_size]
        output = torch.cat(output_list, dim=0)
        # output: [batch, num_classes]
        # output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)
        # output: [batch, 2 * sent_hidden_size]
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state, document_input)
        # output: [batch, num_classes]
        # output = self.fc(output)

        return output


if __name__ == "__main__":
    abc = HierAttDynamicNet(word_hidden_size=128, sent_hidden_size=128, batch_size=128,
                            dict_len=1000, embed_size=300, document_sent_hidden_size=128, device='cuda')
    print(abc)
