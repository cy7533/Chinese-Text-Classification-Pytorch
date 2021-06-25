import os
import pandas as pd
from sklearn.utils import shuffle


"""
输入模型的预处理，改变标签，划分为train dev test
generated_data(用空格分词） -> data(all.csv train.csv ...)
"""


def trans_labels(label):
    """
    改变标签
    :param label:
    :return:
    """
    rules = {
        '天': '0',  # 低（减刑0-3个月）
        '1': '0',
        '2': '0',
        '3': '0',
        '4': '1',  # 中（减刑4-7个月）
        '5': '1',
        '6': '1',
        '7': '1',
        '8': '2',  # 高（减刑8-12个月）
        '9': '2',
        '10': '2',
        '11': '2',
        '12': '2',
        '减去余下': '3',  # 特殊（减去余下、减为无期、减为有期、不予减刑）
        '不予减刑': '3',
        '减为无期': '3',
    }
    return rules.get(label, '3')


def train_dev_test_split(from_csv_file, to_dir, test_size=0.1, is_shuffle=True, random_state=None):
    """

    :param from_csv_file:
    :param to_dir:
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

    data = pd.read_csv(from_csv_file, header=0, index_col=0, keep_default_na=False)
    column_dict = {'result': 'label', 'new_crime': 'crime', 'new_judge': 'judge',
                   'new_performance': 'performance', 'new_fact': 'fact'}
    data = pd.DataFrame(data, columns=column_dict.keys())
    data.rename(columns=column_dict, inplace=True)
    print('已读入数据，总数量：{}，开始处理...'.format(len(data)))

    data['label'] = data['label'].apply(trans_labels)
    print('已转换标签。')

    # 标签统计
    subtype_counts = dict(data['label'].value_counts())
    print(subtype_counts)

    if is_shuffle:
        data = shuffle(data, random_state=random_state)

    train = data[int(len(data) * test_size):].reset_index(drop=True)
    test = data[:int(len(data) * test_size)].reset_index(drop=True)

    print('train no.: {}.'.format(len(train)))
    print('test no.: {}.'.format(len(test)))

    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    train.to_csv(os.path.join(to_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(to_dir, 'dev.csv'), index=False)
    test.to_csv(os.path.join(to_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    # csv_file = 'JXdata/generated_data/20210602/jx_fz_generated.csv'
    #
    # # 原始数据分析
    # df = pd.read_csv(csv_file, header=0, index_col=0, keep_default_na=False)
    # # 标签统计
    # subtype_counts = dict(df['result'].value_counts())
    # print(subtype_counts)
    #
    # train_dev_test_split('JXdata/generated_data/20210602/jx_fz_generated.csv',
    #                      'JXdata/data')

    df = pd.read_csv('JXdata/data/train.csv', header=0, keep_default_na=False)
    subtype_counts1 = dict(df['label'].value_counts())
    print(subtype_counts1)

    df = pd.read_csv('JXdata/data/dev.csv', header=0, keep_default_na=False)
    subtype_counts2 = dict(df['label'].value_counts())
    print(subtype_counts2)
