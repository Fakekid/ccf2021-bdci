# coding:utf-8

import os
import csv
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertForTokenClassification, BertModel
from tqdm import tqdm


def argmax(res):
    k, tmp = 0, 0
    for i in range(len(res)):
        if res[i] > tmp:
            tmp = res[i]
            k = i

    return k


def batch_loader(config, src, seg, mask):
    ins_num = src.size()[0]
    batch_size = config['batch_size']
    for i in range(ins_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        mask_batch = mask[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, seg_batch, mask_batch
    if ins_num > ins_num // batch_size * batch_size:
        src_batch = src[ins_num // batch_size * batch_size:, :]
        seg_batch = seg[ins_num // batch_size * batch_size:, :]
        mask_batch = mask[ins_num // batch_size * batch_size:, :]
        yield src_batch, seg_batch, mask_batch


def read_dataset(config, tokenizer):
    start = time.time()
    dataset = []
    seq_length = config['max_seq_len']

    df = pd.read_csv(config['data_path'])
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sent = row['text']
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
        dataset.append((src, seg, mask))

    print("\n>> loading sentences from {},Time cost:{:.2f}".
          format(config['data_path'], ((time.time() - start) / 60.00)))

    return dataset


def predict(dataset, pre_model, config):
    predict_logits, predictions = [], []

    tags = None

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])
    mask = torch.LongTensor([sample[2] for sample in dataset])

    for i, (src_batch, seg_batch, mask_batch) in \
            enumerate(batch_loader(config, src, seg, mask)):
        src_batch = src_batch.to(config['device'])
        seg_batch = seg_batch.to(config['device'])
        mask_batch = mask_batch.to(config['device'])
        with torch.no_grad():
            output = pre_model(input_ids=src_batch, token_type_ids=seg_batch, attention_mask=mask_batch)

        logits = output[0]
        tag = logits

        tag = torch.softmax(tag, -1)

        tag = np.argmax(tag.cpu().numpy(), -1)

        if tags is None:
            tags = tag
        else:
            tags = np.concatenate([tags, tag], axis=0)

    tags = np.array(tags)

    name2id = {'B-COMMENTS_N': 6,
               'I-PRODUCT': 5,
               'I-COMMENTS_N': 7,
               'B-BANK': 2,
               'B-COMMENTS_ADJ': 8,
               'B-PRODUCT': 4,
               'I-COMMENTS_ADJ': 9,
               'I-BANK': 3,
               'O': 1}

    id2name = {v: k for k, v in name2id.items()}

    mask = mask.cpu().numpy()
    preds = []
    for i in range(len(tags)):
        seq = tags[i]

        if i < 5:
            print([seq[j] for j in range(len(seq)) if mask[i, j] == 1])

        seq = [item if item > 0 else 1 for item in seq]
        seq = [id2name[seq[j]] for j in range(len(seq)) if mask[i, j] == 1][1:-1]  # 去掉序列中的cls和sep
        preds.append([i, ' '.join(seq)])

    result = pd.DataFrame(preds, columns=['id', 'BIO_anno'])
    result.to_csv(os.path.join(config['output_txt_path'], config['output_txt_name']), index=False)
    print('done')

    return predict_logits


def main():
    config = {
        'vocab_path': '',
        'init_model_path': '',
        'data_path': 'data/test_public.csv',
        'load_model_path': 'output_model_tag/checkpoint-9024',
        'output_txt_path': './',
        'output_txt_name': 'predict_tag.csv',
        'batch_size': 128,
        'max_seq_len': 256,
        'device': 'cuda',
    }

    warnings.filterwarnings('ignore')
    start_time = time.time()
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))

    config['vocab_path'] = '/chj/dev/lianxiaolei/model/bert-base-chinese' + '/vocab.txt'

    tokenizer = BertTokenizer.from_pretrained(config['vocab_path'])

    print('读取数据集')
    test_set = read_dataset(config, tokenizer=tokenizer)

    print("\n>> start predict ... ...")

    # model = NeZhaSequenceClassification.from_pretrained(config['load_model_path'])
    print('fine-tune模型({})是否存在：{}'.format(config['load_model_path'], os.path.exists(config['load_model_path'])))
    model = BertForTokenClassification.from_pretrained(config['load_model_path'])
    model.to(config['device'])
    model.eval()

    predict(dataset=test_set, pre_model=model, config=config)

    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at : {}, total cost time : {:.2f}".
          format(localtime_end, (time.time() - start_time) / 60.00))


if __name__ == '__main__':
    main()