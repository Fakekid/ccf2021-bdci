# coding:utf-8

import time
import warnings
from core.trainer import FinetuneTrainer
from core.utils import read_dataset, build_tokenizer


def main():
    with open('conf_tag.txt', 'r', encoding='utf8') as fin:
        c = fin.readlines()
    config = eval(''.join(c))

    warnings.filterwarnings('ignore')
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))
    print("\n>> loading model from :{}".format(config['model_path']))

    tokenizer = build_tokenizer(config['model_path'])

    dataset = read_dataset(config['train_data_path'], tokenizer, config['max_seq_len'], config['task'], 'text', 'tag')

    if config['adv'] == '':
        print('\n>> start normal training ...')
    elif config['adv'] == 'fgm':
        print('\n>> start fgm training ...')
    elif config['adv'] == 'pgd':
        print('\n>> start pgd training ...')

    ft = FinetuneTrainer(epoch=config['num_epochs'], kfold=config['kfold'])
    ft.train(config['model_path'], config['num_labels'], dataset, loss_type=config['loss_type'],
             output_path=config['output_path'], batch_size=config['batch_size'], model_type='tag',
             learning_rate=config['learning_rate'])

    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at:{}".format(localtime_end))


if __name__ == '__main__':
    main()
