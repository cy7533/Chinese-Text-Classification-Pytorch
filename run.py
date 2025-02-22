# coding: UTF-8
import os
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, '
                                                             'TextRNN_Att, DPCNN, Transformer, HANConcat, '
                                                             'HANConcat_Dynamic')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    # dataset = 'JXdata'  # 数据集
    dataset = 'JXPairS'

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'

    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    if model_name == 'HANConcat' or model_name == 'HANConcat_Dynamic':
        from utils_han import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    if model_name == 'HANConcatAlign':
        from utils_align_han import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        if model_name.endswith('Align'):
            model_name = model_name[:-5]
            from utils_align_baseline import build_dataset, build_iterator, get_time_dif
        else:
            from utils_baseline import build_dataset, build_iterator, get_time_dif
    print(model_name)

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    if type(vocab).__name__ == 'dict':
        config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    print(vars(config))
    if not os.path.exists(os.path.dirname(config.save_path)):
        os.makedirs(os.path.dirname(config.save_path))
    with open(os.path.join(os.path.dirname(config.save_path),
                           config.model_name + '_config.txt'), 'a+') as f:
        print(vars(config), file=f)
    torch.save(model.state_dict(), config.save_path)
    train(config, model, train_iter, dev_iter, test_iter)
