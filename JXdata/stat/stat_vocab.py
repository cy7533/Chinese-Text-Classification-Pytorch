"""
用本地 han 环境运行（因为远程服务器上不了网，装不了包。。。）
"""
# 导入必备工具包
import os
import re

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from wordcloud import WordCloud
import pickle as pkl

# 设置显示风格
plt.style.use('fivethirtyeight')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (14, 8)

# 分别读取训练csv和验证csv
all_data = pd.read_csv("../generated_data/20210602/jx_fz_generated.csv", header=0, keep_default_na=False)
train_data = pd.read_csv("../data/train.csv", header=0, keep_default_na=False)
valid_data = pd.read_csv("../data/dev.csv", header=0, keep_default_na=False)

"""
词向量
"""


def get_word_cloud(frequencies):
    # 实例化绘制词云的类, 其中参数font_path是字体路径, 为了能够显示中文,
    # max_words指词云图像最多显示多少个词, background_color为背景颜色
    wordcloud = WordCloud(font_path="./SimHei.ttf", max_words=100, background_color="white")
    # 生成词云
    wordcloud.generate_from_frequencies(frequencies)

    # 绘制图像并显示
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


word_tokenizer = lambda x: x.split(' ')


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


def build_vocab(texts, tokenizer, material_path=None):
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
    return vocab_dic


if __name__ == '__main__':
    # vocab = pkl.load(open('../data/vocab_han.pkl', 'rb'))
    # print(f"Vocab size: crime: {len(vocab.crime)}, judge: {len(vocab.judge)}, performance: {len(vocab.performance)}, "
    #       f"fact: {len(vocab.fact)}")

    # get_word_cloud(vocab.crime)
    # get_word_cloud(vocab.judge)
    # get_word_cloud(vocab.performance)
    # get_word_cloud(vocab.fact)

    crime_vocab = build_vocab(all_data.new_crime.values, word_tokenizer, '../material')
    judge_vocab = build_vocab(all_data.new_judge.values, word_tokenizer, '../material')
    performance_vocab = build_vocab(all_data.new_performance.values, word_tokenizer, '../material')
    fact_vocab = build_vocab(all_data.new_fact.values, word_tokenizer, '../material')

    get_word_cloud(crime_vocab)
    get_word_cloud(judge_vocab)
    get_word_cloud(performance_vocab)
    get_word_cloud(fact_vocab)

