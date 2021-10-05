# coding: utf8

import random
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def batch_loader(config, src, tgt, seg, mask):
    ins_num = src.size()[0]
    batch_size = config['batch_size']
    for i in range(ins_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        mask_batch = mask[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch, mask_batch
    if ins_num > ins_num // batch_size * batch_size:
        src_batch = src[ins_num // batch_size * batch_size:, :]
        tgt_batch = tgt[ins_num // batch_size * batch_size:]
        seg_batch = seg[ins_num // batch_size * batch_size:, :]
        mask_batch = mask[ins_num // batch_size * batch_size:, :]
        yield src_batch, tgt_batch, seg_batch, mask_batch


def read_dataset(config, tokenizer, return_len=False, mode='train', shuffle=True):
    dataset = []
    seq_length = config['max_seq_len']
    task = config['task']

    df = pd.read_csv(config['data_path'])
    length = df.shape[0]

    x_col = config['x_col_name']
    y_col = config['y_col_name']

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sent = row[x_col]
        src = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(sent) + ['SEP'])
        seg = [0] * len(src)
        mask = [1] * len(src)

        if len(src) > seq_length:
            src = src[: seq_length]
            seg = seg[: seq_length]
            mask = mask[: seq_length]

        while len(src) < seq_length:
            src.append(0)
            seg.append(0)
            mask.append(0)

        if mode == 'train':
            label = row[y_col]
            if task == 'tag' or task == 'tagging' or task == 'seq':
                label = label.split(' ')
                label = [int(1)] + list(map(int, label)) + [int(1)]
            else:
                label = int(label)
            dataset.append((src, seg, mask, label))
        else:
            dataset.append((src, seg, mask))

    if mode == 'train':
        data = pd.DataFrame(dataset, columns=['input_ids', 'segment_ids', 'input_mask', 'label'])
    else:
        data = pd.DataFrame(dataset, columns=['input_ids', 'segment_ids', 'input_mask'])

    if shuffle:
        np.random.shuffle(data)

    if return_len:
        return data, length
    return data


def read_dataset(data_path, tokenizer, max_seq_len=128, test_line=None):
    dataset = []
    test_dataset = []
    seq_length = max_seq_len

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(tqdm(f)):
            if line_id == 0: continue
            sent, tag, tgt = line.strip().split('\t')
            src = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(sent) + ['SEP'])

            seg = [0] * len(src)
            mask = [1] * len(src)

            if len(src) > seq_length:
                src = src[: seq_length]
                seg = seg[: seq_length]
                mask = mask[: seq_length]
            while len(src) < seq_length:
                src.append(0)
                seg.append(0)
                mask.append(0)

            if test_line is not None:
                if line_id in test_line:
                    test_dataset.append((src, int(tgt), seg, mask))
                else:
                    dataset.append((src, int(tgt), seg, mask))
            else:
                dataset.append((src, int(tgt), seg, mask))

    if test_line is not None:
        return dataset, test_dataset
    else:
        return dataset


def block_shuffle(config, train_set):
    bs = config['batch_size'] * 100
    num_block = int(len(train_set) / bs)
    slice_ = num_block * bs

    train_set_to_shuffle = train_set[:slice_]
    train_set_left = train_set[slice_:]

    sorted_train_set = sorted(train_set_to_shuffle, key=lambda i: len(i[0]))
    shuffled_train_set = []

    tmp = []
    for i in range(len(sorted_train_set)):
        tmp.append(sorted_train_set[i])
        if (i + 1) % bs == 0:
            random.shuffle(tmp)
            shuffled_train_set.extend(tmp)
            tmp = []

    random.shuffle(train_set_left)
    shuffled_train_set.extend(train_set_left)

    return shuffled_train_set
