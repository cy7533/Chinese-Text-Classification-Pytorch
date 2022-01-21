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

# 分别读取训练csv和验证csv
all_data = pd.read_csv("../generated_data/20210602/jx_fz_generated.csv", header=0, keep_default_na=False)
train_data = pd.read_csv("../data/train.csv", header=0, keep_default_na=False)
valid_data = pd.read_csv("../data/dev.csv", header=0, keep_default_na=False)


"""
句子长度
"""

# crime
# 在训练数据中添加新的句子长度列, 每个元素的值都是对应的句子列的长度
all_data["crime_length"] = list(map(lambda x: len(x.split(' ')), all_data["new_crime"]))

# 绘制句子长度列的数量分布图
sns.countplot("crime_length", data=all_data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
# plt.xticks([])
plt.show()

# # 绘制dist长度分布图
# sns.distplot(all_data["crime_length"])
# # 主要关注dist长度分布横坐标, 不需要绘制纵坐标
# plt.yticks([])
# plt.show()

# judge
# 在训练数据中添加新的句子长度列, 每个元素的值都是对应的句子列的长度
all_data["judge_length"] = list(map(lambda x: len(x), all_data["judge"]))

# 绘制句子长度列的数量分布图
sns.countplot("judge_length", data=all_data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
plt.show()

# 绘制dist长度分布图
sns.distplot(all_data["judge_length"])
# 主要关注dist长度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()

# performance
# 在训练数据中添加新的句子长度列, 每个元素的值都是对应的句子列的长度
all_data["performance_length"] = list(map(lambda x: len(x), all_data["performance"]))

# 绘制句子长度列的数量分布图
sns.countplot("performance_length", data=all_data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
plt.show()

# 绘制dist长度分布图
sns.distplot(all_data["performance_length"])
# 主要关注dist长度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()

# fact
# 在训练数据中添加新的句子长度列, 每个元素的值都是对应的句子列的长度
all_data["fact_length"] = list(map(lambda x: len(x), all_data["fact"]))

# 绘制句子长度列的数量分布图
sns.countplot("fact_length", data=all_data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
plt.show()

# 绘制dist长度分布图
sns.distplot(all_data["fact_length"])
# 主要关注dist长度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()
