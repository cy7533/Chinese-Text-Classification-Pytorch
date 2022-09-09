import numpy as np
import torch
import torch.nn as nn

from models.HAN.hierarchical_att_model import HierAttNet
from models.HAN.word_att_model import WordAttNet

"""
crime + judge + performance 并联  || article -> 文本对齐（0 or 1）
无 dynamic attention 版本
"""


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'HANConcatAlign'

        self.train_path = dataset + '/data/train.csv'  # 训练集
        self.dev_path = dataset + '/data/dev.csv'  # 验证集
        self.test_path = dataset + '/data/test.csv'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.material_path = '/home/chiyao/projects/HANpytorch/material/'  # 数据材料文件夹（停用词，词典）

        self.vocab_path = dataset + '/data/vocab_han.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.8  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数

        self.crime_n_vocab = 0  # crime词表大小，在运行时赋值
        self.judge_n_vocab = 0  # judge词表大小，在运行时赋值
        self.performance_n_vocab = 0  # performance词表大小，在运行时赋值
        self.fact_n_vocab = 0  # fact词表大小，在运行时赋值
        self.article_n_vocab = 0  # article词表大小，在运行时赋值

        self.crime_max_length = 0  # crime取最大数，在运行时赋值
        self.judge_max_length = (0, 0)  # judge取最大的词、句数，在运行时赋值
        self.performance_max_length = (0, 0)  # performance取最大的词、句数，在运行时赋值
        self.fact_max_length = (0, 0)  # fact取最大的词、句数，在运行时赋值
        self.article_max_length = (0, 0)  # article取最大的词、句数，在运行时赋值

        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一

        self.crime_hidden_size = 128  # crime gru隐藏层
        self.judge_word_hidden_size = 128  # judge gru隐藏层
        self.judge_sent_hidden_size = 128  # judge gru隐藏层
        self.performance_word_hidden_size = 128  # performance gru隐藏层
        self.performance_sent_hidden_size = 128  # performance gru隐藏层
        self.fact_word_hidden_size = 128  # fact gru隐藏层
        self.fact_sent_hidden_size = 128  # fact gru隐藏层
        self.article_word_hidden_size = 128  # fact gru隐藏层
        self.article_sent_hidden_size = 128  # fact gru隐藏层

        self.num_epochs = 100  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.batch_size = config.batch_size
        self.crime_hidden_size = config.crime_hidden_size
        self.device = config.device

        self.crime_encoder = WordAttNet(config.crime_n_vocab, config.embed, config.crime_hidden_size)
        self.judge_encoder = HierAttNet(config.judge_word_hidden_size, config.judge_sent_hidden_size,
                                        config.batch_size, config.judge_n_vocab, config.embed, config.device)
        self.performance_encoder = HierAttNet(config.performance_word_hidden_size, config.performance_sent_hidden_size,
                                              config.batch_size, config.performance_n_vocab, config.embed,
                                              config.device)
        self.fact_encoder = HierAttNet(config.fact_word_hidden_size, config.fact_sent_hidden_size,
                                       config.batch_size, config.fact_n_vocab, config.embed, config.device)
        self.article_encoder = HierAttNet(config.article_word_hidden_size, config.article_sent_hidden_size,
                                          config.batch_size, config.article_n_vocab, config.embed, config.device)

        self.fc = nn.Linear(2 * (config.crime_hidden_size +
                                 config.judge_sent_hidden_size +
                                 config.performance_sent_hidden_size), 2 * config.article_sent_hidden_size)

        self.dropout = nn.Dropout(p=config.dropout)

        self.fc_out = nn.Linear(4 * config.article_sent_hidden_size, config.num_classes)
        self.dropout_out = nn.Dropout(p=config.dropout)

        self.crime_hidden_state = None
        self.init_hidden_state()

    def init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.crime_hidden_state = torch.randn(2, batch_size, self.crime_hidden_size)
        if torch.cuda.is_available():
            self.crime_hidden_state = self.crime_hidden_state.to(self.device)
        self.judge_encoder.init_hidden_state(last_batch_size)
        self.performance_encoder.init_hidden_state(last_batch_size)
        self.fact_encoder.init_hidden_state(last_batch_size)
        self.article_encoder.init_hidden_state(last_batch_size)

    def forward(self, x):
        """
        crime_input: [batch, crime_max_length]
        judge_input: [batch, judge_max_sent_length, judge_max_word_length]
        performance_input: [batch, performance_max_sent_length, performance_max_word_length]
        fact_input: [batch, fact_max_sent_length, fact_max_word_length]
        fact_input: [batch, article_max_sent_length, article_max_word_length]
        """
        crime_input = x['crime']
        judge_input = x['judge']
        performance_input = x['performance']
        fact_input = x['fact']
        article_input = x['article']

        # crime_input: [crime_max_length, batch]
        crime_input = crime_input.permute(1, 0)
        # print(crime_input.shape)
        # crime_output: [1, batch, 2 * crime_hidden_size]
        crime_output, self.crime_hidden_state = self.crime_encoder(crime_input, self.crime_hidden_state)
        # crime_output: [batch, 2 * crime_hidden_size]
        crime_output = crime_output.squeeze(0)

        # judge_output: [batch, 2 * judge_sent_hidden_size]
        judge_output = self.judge_encoder(judge_input)

        # performance_output: [batch, 2 * performance_sent_hidden_size]
        performance_output = self.performance_encoder(performance_input)

        # fact_output: [batch, 2 * fact_sent_hidden_size]
        # fact_output = self.fact_encoder(fact_input)

        article_output = self.article_encoder(article_input)

        # concat_output: [batch, 2 * (self.crime_hidden_size + self.judge_sent_hidden_size +
        # self.performance_sent_hidden_size)]
        concat_output = torch.cat((crime_output, judge_output, performance_output), dim=1)

        concat_output = self.dropout(concat_output)

        # output: [batch, 2 * self.article_sent_hidden_size]
        concat_output = self.fc(concat_output)

        output = torch.cat((concat_output, article_output), dim=1)

        output = self.dropout_out(output)

        output = self.fc_out(output)

        return output
