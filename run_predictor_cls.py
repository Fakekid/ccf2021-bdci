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
from core.modules import BertForSequenceClassification
from tqdm import tqdm
from core.utils import build_tokenizer


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


def predict(dataset, config, is_cv=False):
    predict_logits = []

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])
    mask = torch.LongTensor([sample[2] for sample in dataset])

    if not is_cv:
        model = BertForSequenceClassification.from_pretrained(
            os.path.join(config['load_model_path'], 'finetune_model'))
        model.to(config['device'])
        model.eval()

        cls = None
        for i, (src_batch, seg_batch, mask_batch) in \
                enumerate(batch_loader(config, src, seg, mask)):
            src_batch = src_batch.to(config['device'])
            seg_batch = seg_batch.to(config['device'])
            mask_batch = mask_batch.to(config['device'])
            with torch.no_grad():
                output = model(input_ids=src_batch, token_type_ids=seg_batch, attention_mask=mask_batch)

            logits = output[0]
            logits = torch.softmax(logits, dim=-1)
            logits = logits.cpu().numpy()

            if cls is None:
                cls = logits
            else:
                cls = np.concatenate([cls, logits], axis=0)
        predict_logits = cls
    else:

        for fold in tqdm(range(config['kfold'])):
            torch.cuda.empty_cache()
            print('load model from {}'.format(config['load_model_path'] + f'_fold{fold}'))
            model = BertForSequenceClassification.from_pretrained(
                os.path.join(config['load_model_path'] + f'_fold{fold}', 'finetune_model'))
            model.to(config['device'])
            model.eval()

            cls = None
            for i, (src_batch, seg_batch, mask_batch) in \
                    enumerate(batch_loader(config, src, seg, mask)):
                src_batch = src_batch.to(config['device'])
                seg_batch = seg_batch.to(config['device'])
                mask_batch = mask_batch.to(config['device'])
                with torch.no_grad():
                    output = model(input_ids=src_batch, token_type_ids=seg_batch, attention_mask=mask_batch)

                logits = output[0]

                logits = logits.cpu().numpy()

                if cls is None:
                    cls = logits
                else:
                    cls = np.concatenate([cls, logits], axis=0)

            predict_logits.append(cls)

    predict_logits = np.array(predict_logits)

    if predict_logits.ndim == 3:
        predict_logits = np.mean(predict_logits, axis=0)

    # weight = np.array([3.23984015, 4.25029316, 1.37903608])
    # predict_logits = predict_logits * weight

    np.save('pred_logits', predict_logits)

    cls = np.argmax(np.array(predict_logits), axis=-1)

    id_ = np.array(list(range(len(cls))))
    cls = np.array(cls)

    result = pd.DataFrame(np.concatenate([id_.reshape([-1, 1]), cls.reshape([-1, 1])], axis=1), columns=['id', 'class'])
    result.to_csv(os.path.join(config['output_txt_path'], config['output_txt_name']), index=False)
    print('done')

    return predict_logits


def main():
    with open('conf_cls_dev.txt', 'r', encoding='utf8') as fin:
        c = fin.readlines()
    config = eval(''.join(c))

    warnings.filterwarnings('ignore')
    start_time = time.time()
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))

    tokenizer = build_tokenizer(config['vocab_path'])

    print('读取数据集')
    test_set = read_dataset(config, tokenizer=tokenizer)

    print("\n>> start predict ... ...")
    predict(dataset=test_set, config=config)

    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at : {}, total cost time : {:.2f}".
          format(localtime_end, (time.time() - start_time) / 60.00))


if __name__ == '__main__':
    main()
