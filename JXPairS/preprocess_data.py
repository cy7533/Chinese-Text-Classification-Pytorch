import os
import pandas as pd
from sklearn.utils import shuffle

"""
法条对数据
来源：/home/chiyao/projects/HANpytorch/article_data（linux服务器）
/Users/cheryl/Documents/WorkFolder/MyProjects/智慧司法项目/HANpytorch/article_data（mac本地）

版本：采用20220117目录下的数据

输入模型的预处理，改变标签，划分为train dev test
采用20220117目录下的以法条名称和号码为文件名的所有文件（每个文件都有18968条数据）(用空格分词） 
-> data(all.csv train.csv ...) （大类部分随机抽取 负采样）
"""


def train_dev_test_split(data, test_size=0.1, is_shuffle=True, random_state=None):
    """

    :param data: dataframe
    :param test_size:float
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.
    :param is_shuffle: boolean, optional (default=None)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    :param random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :return:
    """
    if is_shuffle:
        data = shuffle(data, random_state=random_state)

    train = data[int(len(data) * test_size):].reset_index(drop=True)
    test = data[:int(len(data) * test_size)].reset_index(drop=True)

    print('train no.: {}.'.format(len(train)))
    print('test no.: {}.'.format(len(test)))

    return train, test


def read_articles_stat(path):
    """
    读取articles_stat文件，生成可以读取法条正负样本数量的字典列表
    :param path: articles_stat文件位置
    :return: ret: [{'第一条': (负样本数量, 正样本数量), ...}, {...}, ..., {...}]
    """
    df = pd.read_csv(path, header=0, index_col=0)
    ret = [{} for i in range(4)]
    for index, row in df.iterrows():
        ret[row['name']][row['number']] = (row['negatives'], row['positives'])
    # print(ret)
    return ret


def get_all_sample(from_dir, to_dir, test_size=0.1, is_shuffle=True, random_state=None):
    column_dict = {'new_crime': 'crime', 'new_judge': 'judge',
                   'new_performance': 'performance', 'new_fact': 'fact',
                   'new_article': 'article', 'catch': 'label'}
    train_list = []
    test_list = []

    for filename in os.listdir(from_dir):
        if 'pairs.csv' in filename:
            # article_args = filename.split('_')
            # article_name = int(article_args[0])
            # article_number = article_args[1]
            df = pd.read_csv(os.path.join(from_dir, filename), header=0, index_col=0, keep_default_na=False)
            df = pd.DataFrame(df, columns=column_dict.keys())
            df.rename(columns=column_dict, inplace=True)
            negatives_df = df[df['label'] == 0]
            positives_df = df[df['label'] == 1]
            negative_num = len(negatives_df)
            positive_num = len(positives_df)
            print('已读入{}数据，总数量：{}，正例数：{}，负例数：{}，开始处理...'.
                  format(filename, len(df), negative_num, positive_num))
            if negative_num < positive_num:
                if is_shuffle:
                    positives_df = shuffle(positives_df, random_state=random_state)
                positives_df = positives_df[:negative_num]
            elif positive_num < negative_num:
                if is_shuffle:
                    negatives_df = shuffle(negatives_df, random_state=random_state)
                negatives_df = negatives_df[:positive_num]

            if is_shuffle:
                positives_df = shuffle(positives_df, random_state=random_state)
                negatives_df = shuffle(negatives_df, random_state=random_state)

            print('{}的正样本分割：'.format(filename))
            positives_train, positives_test = train_dev_test_split(positives_df, test_size, is_shuffle, random_state)
            print('{}的负样本分割：'.format(filename))
            negatives_train, negatives_test = train_dev_test_split(negatives_df, test_size, is_shuffle, random_state)
            train_list.append(positives_train)
            train_list.append(negatives_train)
            test_list.append(positives_test)
            test_list.append(negatives_test)
    train = pd.concat(train_list)
    test = pd.concat(test_list)
    print('总计：')
    print('all train no.: {}.'.format(len(train)))
    print('all test no.: {}.'.format(len(test)))

    if is_shuffle:
        train = shuffle(train, random_state=random_state)
        test = shuffle(test, random_state=random_state)

    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    train.to_csv(os.path.join(to_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(to_dir, 'dev.csv'), index=False)
    test.to_csv(os.path.join(to_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    from_dir = '/home/chiyao/projects/HANpytorch/article_data/20220117'
    # articles_stat = read_articles_stat(os.path.join(from_dir, 'articles_stat.csv'))
    get_all_sample(from_dir, to_dir='data', is_shuffle=True, random_state=0)
