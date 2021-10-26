import torch

"""
HAN模型计算工具集
"""


def matrix_mul(x, weight, bias=False):
    """
    tanh(weight * x + bias)
    :param x: [feature_size, m1, m2]
    :param weight: [m2, m3]
    :param bias:
    """
    feature_list = []
    for feature in x:
        # feature: [m1, m3]
        feature = torch.mm(feature, weight)
        # print('torch.mm:', torch.isnan(feature).int().sum())
        # print('torch.mm:', feature)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        # feature: [1, m1, m3]
        feature = torch.tanh(feature).unsqueeze(0)
        # print('torch.tanh:', torch.isnan(feature).int().sum())
        # print('torch.tanh:', feature)
        # feature_list: feature_size * [1, m1, m3]
        feature_list.append(feature)

    # [feature_size, m1, m3]
    return torch.cat(feature_list, 0)


def element_wise_mul(input1, input2):
    """
    矩阵点乘
    :param input1: [feature_size, m1, m2]
    :param input2: [feature_size, m1]
    """
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        # feature_2: [m1, 1] => [m1, m2]
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        # feature: [m1, m2]
        feature = feature_1 * feature_2
        # feature_list: feature_size * [1, m1, m2]
        feature_list.append(feature.unsqueeze(0))
    # output: [feature_size, 1, m1, m2]
    output = torch.cat(feature_list, 0)

    # [1, 1, m1, m2] => [m1, m2]
    return torch.sum(output, 0).unsqueeze(0)


def context_matrix_mul(x, weight):
    """
    动态context vector 点乘计算
    :param x: [feature_size, m1, m2]
    :param weight: [m2, m1]
    """
    feature_list = []
    for feature in x:
        # feature: [m1, m2] => [m1, m1]
        feature = torch.mm(feature, weight)
        # 取对角线的位置
        # feature: [m1] => [1, m1]
        feature = torch.diag(feature, 0).unsqueeze(0)
        # feature_list: feature_size * [1, m1]
        feature_list.append(feature)

    # [feature_size, 1, m1] => [feature_size, m1]
    return torch.stack(feature_list).squeeze(1)
