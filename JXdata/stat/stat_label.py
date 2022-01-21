"""
用本地 han 环境运行（因为远程服务器上不了网，装不了包。。。）
"""
# 导入必备工具包
import re

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

# 设置显示风格
plt.style.use('fivethirtyeight')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (14, 8)

"""
标签
"""

# 分别读取训练csv和验证csv
all_data = pd.read_csv("./generated_data/20210602/jx_fz_generated.csv", header=0, keep_default_na=False)
train_data = pd.read_csv("./data/train.csv", header=0, keep_default_na=False)
valid_data = pd.read_csv("./data/dev.csv", header=0, keep_default_na=False)


def trans_labels(label):
    """
    改变标签
    :param label:
    :return:
    """
    rules = {
        '天': '天',  # 低（减刑0-3个月）
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,  # 中（减刑4-7个月）
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,  # 高（减刑8-12个月）
        '9': 9,
        '10': 10,
        '11': 11,
        '12': 12,
        '减去余下': '减去余下',  # 特殊（减去余下、减为无期、减为有期、不予减刑）
        '不予减刑': '不予减刑',
        '减为无期': '减为无期',
    }
    return rules.get(label, '其他')


label_order = ['天', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, '减去余下', '不予减刑', '减为无期', '其他']

# 获得总数据标签数量分布
# 转换标签
all_data['result'] = all_data['result'].apply(trans_labels)
print('已转换标签。')
sns.countplot("result", data=all_data, order=label_order)
plt.title("data")
plt.show()

# 获得训练数据标签数量分布
sns.countplot("label", data=train_data)
plt.title("train_data")
plt.show()

# 获取验证数据标签数量分布
sns.countplot("label", data=valid_data)
plt.title("valid_data")
plt.show()

