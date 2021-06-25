# coding: UTF-8
import os
import re

import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from collections import namedtuple

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


def build_vocab(texts, tokenizer, max_size, min_freq, material_path=None):
    """
    构建词典
    :param texts: [text, text, ...]
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

    vocab_dic = {}
    for text in tqdm(texts):
        text = text.strip()
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

VocabTuple = namedtuple('VocabTuple', ['crime', 'judge', 'performance', 'fact'])


def sent_tokenizer(para):
    """
    中文分割句子
    """
    para = re.sub(r'([。.，,！!？?；;])([^”’"\'])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub(r'(\\.{6})([^”’"])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(\\…{2})([^”’"])', r"\1\n\2", para)  # 中文省略号
    # 如果双引号前有终止符，那么双引号才是句子的终点，
    para = re.sub(r'([。.，,！!？?；;][”’"\'])([^。.，,！!？?；;])', r'\1\n\2', para)
    return [sen.strip() for sen in para.split("\n")]


def build_dataset(config, use_word):
    if use_word:
        word_tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        word_tokenizer = lambda x: [y for y in x]  # char-level

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        train_df = pd.read_csv(config.train_path, header=0, keep_default_na=False)
        print('Building crime_vocab...')
        crime_vocab = build_vocab(train_df.crime.values, word_tokenizer, MAX_VOCAB_SIZE, 1, config.material_path)
        print('crime_vocab', crime_vocab)
        print('Building judge_vocab...')
        judge_vocab = build_vocab(train_df.judge.values, word_tokenizer, MAX_VOCAB_SIZE, 1, config.material_path)
        print('judge_vocab', judge_vocab)
        print('Building performance_vocab...')
        performance_vocab = build_vocab(train_df.performance.values, word_tokenizer, MAX_VOCAB_SIZE, 1,
                                        config.material_path)
        print('performance_vocab', performance_vocab)
        print('Building fact_vocab...')
        fact_vocab = build_vocab(train_df.fact.values, word_tokenizer, MAX_VOCAB_SIZE, 1, config.material_path)
        print('fact_vocab', fact_vocab)
        vocab = VocabTuple(crime=crime_vocab, judge=judge_vocab, performance=performance_vocab, fact=fact_vocab)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: crime: {len(vocab.crime)}, judge: {len(vocab.judge)}, performance: {len(vocab.performance)}, "
          f"fact: {len(vocab.fact)}")

    if config.crime_n_vocab == 0:
        config.crime_n_vocab = len(vocab.crime)
        config.judge_n_vocab = len(vocab.judge)
        config.performance_n_vocab = len(vocab.performance)
        config.fact_n_vocab = len(vocab.fact)

    def get_max_lengths(data_list, vocab):
        """
        获得句子和单词的80%位置的长度
        param data_list: 数据集python列表
        """

        word_length_list = []
        sent_length_list = []

        for idx, text in enumerate(data_list):
            sent_list = sent_tokenizer(text)
            sent_length_list.append(len(sent_list))
            for sent in sent_list:
                word_list = word_tokenizer(sent)
                word_list_no_stop = []
                for word in word_list:
                    if word in vocab:
                        word_list_no_stop.append(word)
                word_length_list.append(len(word_list_no_stop))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)
        return sorted_word_length[int(0.8 * len(sorted_word_length))], sorted_sent_length[
            int(0.8 * len(sorted_sent_length))]

    if config.crime_max_length == 0:
        train_df = pd.read_csv(config.train_path, header=0, keep_default_na=False)
        config.crime_max_length = 3
        config.judge_max_length = get_max_lengths(train_df.judge.values, vocab.judge)
        config.performance_max_length = get_max_lengths(train_df.performance.values, vocab.performance)
        config.fact_max_length = get_max_lengths(train_df.fact.values, vocab.fact)

        print('crime_max_length', config.crime_max_length)
        print('judge_max_length', config.judge_max_length)
        print('performance_max_length', config.performance_max_length)
        print('fact_max_length', config.fact_max_length)

    def encode_document(text, vocab, max_length):
        """
        文本转为字典中的序列
        param text: 一段文本
        """
        max_length_words, max_length_sentences = max_length
        document_encode = [
            [vocab.get(word, vocab.get(UNK)) for word in word_tokenizer(sentence)]
            for sentence in sent_tokenizer(text)
        ]

        for sentence in document_encode:
            if len(sentence) < max_length_words:
                extended_words = [0 for _ in range(max_length_words - len(sentence))]
                sentence.extend(extended_words)

        if len(document_encode) < max_length_sentences:
            extended_sentences = [
                [0 for _ in range(max_length_words)]
                for _ in range(max_length_sentences - len(document_encode))
            ]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:max_length_words] for sentences in document_encode][:max_length_sentences]
        document_encode = np.stack(arrays=document_encode, axis=0)
        return document_encode.astype(np.int32)

    def encode_text(text, vocab, max_length_words):
        document_encode = [vocab.get(word, vocab.get(UNK)) for word in word_tokenizer(text)]
        if len(document_encode) < max_length_words:
            extended_words = [0 for _ in range(max_length_words - len(document_encode))]
            document_encode.extend(extended_words)
        document_encode = document_encode[:max_length_words]
        document_encode = np.stack(arrays=document_encode, axis=0)
        return document_encode.astype(np.int32)

    def load_dataset(path):
        contents = []
        df = pd.read_csv(path, header=0, keep_default_na=False)

        for i, row in tqdm(df.iterrows()):
            crime_encode = encode_text(row['crime'], vocab.crime, config.crime_max_length)
            judge_encode = encode_document(row['judge'], vocab.judge, config.judge_max_length)
            performance_encode = encode_document(row['performance'], vocab.performance, config.performance_max_length)
            fact_encode = encode_document(row['fact'], vocab.fact, config.fact_max_length)
            contents.append({
                'crime': crime_encode,
                'judge': judge_encode,
                'performance': performance_encode,
                'fact': fact_encode,
                'label': row['label']
            })
        return contents

    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)

    print('train dataset len:', len(train))
    print('dev dataset len:', len(dev))
    print('test dataset len:', len(test))
    return vocab, train, dev, test


"""
3. 构建数据生成器
"""


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

    def _to_tensor(self, data):
        x_crime = torch.LongTensor([_['crime'] for _ in data]).to(self.device)
        x_judge = torch.LongTensor([_['judge'] for _ in data]).to(self.device)
        x_performance = torch.LongTensor([_['performance'] for _ in data]).to(self.device)
        x_fact = torch.LongTensor([_['fact'] for _ in data]).to(self.device)
        y = torch.LongTensor([_['label'] for _ in data]).to(self.device)

        return {
                   'crime': x_crime,
                   'judge': x_judge,
                   'performance': x_performance,
                   'fact': x_fact,
                   'last_batch_size': None
               }, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            batches = self._to_tensor(batches)
            batches[0]['last_batch_size'] = len(self.batches) - self.index * self.batch_size
            self.index += 1
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batches = self._to_tensor(batches)
            self.index += 1
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
