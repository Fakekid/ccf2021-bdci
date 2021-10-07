# coding:utf-8

import time
import warnings

from transformers import (
    BertTokenizer,
    BertTokenizerFast,
)

from core.trainer import FinetuneTrainer
from core.utils import read_dataset


def build_tokenizer(config):
    tokenizer_path = config['model_path'] + '/vocab.txt'
    if config['tokenizer_fast']:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    return tokenizer


# def log_sum_exp(vec):
#     def argmax():
#         # return the argmax as a python int
#         # 返回vec的dim为1维度上的最大值索引
#         _, idx = torch.max(vec, 1)
#         return idx.item()
#
#     max_score = vec[0, argmax()]
#     max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
#     return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def main():
    config = {
        'num_labels': 10,
        'use_model': 'bert',
        'train_data_path': 'data/data.csv',
        # 'dev_data_path': 'data/test.csv',
        'output_path': 'output_model_tag',
        # 'model_path': '/chj/dev/lianxiaolei/model/bert-base-chinese',  # your pretrain model path
        'model_path': '/chj/dev/lianxiaolei/bert_pretrain/mlm_model_prod_opin',  # your pretrain model path
        'shuffle_way': '',
        'use_swa': True,
        'tokenizer_fast': False,
        'task': 'tag',
        'batch_size': 64,
        'num_epochs': 5,
        'max_seq_len': 256,
        'kfold': 5,
        'learning_rate': 2e-5,
        'alpha': 0.3,
        'epsilon': 1.0,
        'adv_k': 3,
        'emb_name': 'word_embeddings.',
        'adv': '',
        'warmup_ratio': 0.05,
        'weight_decay': 0.01,
        'device': 'cuda',
        'seed': 2022
    }

    warnings.filterwarnings('ignore')
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))
    print("\n>> loading model from :{}".format(config['model_path']))

    tokenizer = build_tokenizer(config)

    # data_path, tokenizer, max_seq_len, task, x_col, y_col, return_len=False, mode='train', shuffle=True
    dataset = read_dataset(config['train_data_path'], tokenizer, config['max_seq_len'], config['task'], 'text', 'tag')

    if config['adv'] == '':
        print('\n>> start normal training ...')
    elif config['adv'] == 'fgm':
        print('\n>> start fgm training ...')
    elif config['adv'] == 'pgd':
        print('\n>> start pgd training ...')

    ft = FinetuneTrainer(epoch=config['num_epochs'], kfold=config['kfold'])
    ft.train(config['model_path'], 10, dataset, x_col='text', y_col='tag', output_path=config['output_path'],
             batch_size=config['batch_size'], model_type='tag', learning_rate=2e-5)

    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at:{}".format(localtime_end))


if __name__ == '__main__':
    main()
