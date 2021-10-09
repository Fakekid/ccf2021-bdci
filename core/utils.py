# coding: utf8

import random
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.utils import shuffle


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def batch_loader(batch_size, src, tgt, seg, mask):
    ins_num = src.size()[0]

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


def read_dataset(data_path, tokenizer, max_seq_len, task, x_col, y_col, return_len=False, mode='train',
                 is_shuffle=True):
    dataset = []
    seq_length = max_seq_len

    df = pd.read_csv(data_path, sep='\t')
    length = df.shape[0]

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
                if len(label) > seq_length:
                    label = label[:seq_length]
                while len(label) < seq_length:
                    label.append(0)
            else:
                label = int(label)
            dataset.append((src, seg, mask, label))
        else:
            dataset.append((src, seg, mask))

    if mode == 'train':
        data = pd.DataFrame(dataset, columns=['input_ids', 'segment_ids', 'input_mask', 'label'])
    else:
        data = pd.DataFrame(dataset, columns=['input_ids', 'segment_ids', 'input_mask'])

    if is_shuffle:
        data = shuffle(data)

    if return_len:
        return data, length
    return data
