# coding: UTF-8
import os

import pandas as pd
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

"""
构建词典 vocab
构建数据集 dataset
构建数据生成器
"""

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

"""
1. 构建词典 vocab
"""


def get_stopwords(stopwords_path):
    """
    获取停用词列表
    :param stopwords_path: 停用词数据文件位置
    :return: list
    """
    stopwords = list()
    with open(stopwords_path, 'r') as f:
        for row in f.readlines():
            stopwords.append(row.strip())
    return stopwords


def build_vocab(file_path, tokenizer, max_size, min_freq, material_path=None):
    """
    构建词典
    :param file_path: csv文件路径
    :param tokenizer: 划分函数
    :param max_size: 词典最大词数
    :param min_freq: 保留的最小频次
    :param material_path: 停用词所在的路径，存在则去除停用词
    :return:
    """
    stopwords = []
    if material_path:
        stopwords = get_stopwords(os.path.join(material_path, 'hit_stopwords.txt'))
        stopwords = stopwords + get_stopwords(os.path.join(material_path, 'judge_stopwords.txt'))
        stopwords = stopwords + get_stopwords(os.path.join(material_path, 'performance_stopwords.txt'))

    df = pd.read_csv(file_path, header=0, keep_default_na=False)

    vocab_dic = {}
    for i, row in tqdm(df.iterrows()):
        text = row['crime'] + ' ' + row['judge'] + ' ' + row['performance'] + row['fact']
        if not text:
            continue
        for word in tokenizer(text):
            if word in stopwords:  # 去除停用词
                continue
            vocab_dic[word] = vocab_dic.get(word, 0) + 1  # 出现频次
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                 :max_size]  # 出现频次从大到小排列，截取到最大长度max_size
    vocab_dic = {word_count[0]: idx + 1 for idx, word_count in enumerate(vocab_list)}  # 单词数字范围：[1...len(vocab_dic)]
    vocab_dic.update({UNK: len(vocab_dic) + 1, PAD: 0})
    return vocab_dic


"""
2. 构建数据集 dataset
"""


def build_dataset(config, use_word):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer, MAX_VOCAB_SIZE, 1, config.material_path)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=64):
        contents = []
        df = pd.read_csv(path, header=0, keep_default_na=False)

        for i, row in tqdm(df.iterrows()):
            content = row['crime'] + ' ' + row['judge'] + ' ' + row['performance'] + row['fact']
            label = row['label']
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            token = tokenizer(content)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
