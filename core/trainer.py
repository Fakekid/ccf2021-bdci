# coding:utf8

import os
import numpy as np
import torch
from torch.optim import AdamW
from torch.backends import cudnn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from prettytable import PrettyTable
from core.modules import BertForSequenceClassification, BertForTokenClassification
from core.optimizer import WarmupLinearSchedule
from core.utils import batch_loader
from core.metrics import multi_cls_metrics


def build_model(ptm_name, num_labels, type='cls'):
    if type == 'cls':
        model = BertForSequenceClassification.from_pretrained(ptm_name, num_labels=num_labels)
    elif type == 'tag':
        model = BertForTokenClassification.from_pretrained(ptm_name, num_labels=num_labels)
    else:
        raise ValueError('暂不支持cls和tag以外的模型')
    return model


def build_optimizer(model, train_steps, learning_rate, weight_decay, warmup_ratio):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * warmup_ratio,
                                     t_total=train_steps)

    return optimizer, scheduler


class FinetuneTrainer:
    """

    """

    def __init__(self, **kwargs):
        self.kfold = 1
        self.epoch = 10
        for k, v in kwargs.items():
            print(f'设置 {k}={v}')
            exec(f'self.{k} = {v}')

    def train(self, ptm_name, num_labels, data, output_path, batch_size=128, model_type='cls',
              device='cuda',
              weight_decay=0.01, learning_rate=1e-5, warmup_ratio=0.1):
        """
        训练模型
        """
        kfold = self.kfold
        epoch = self.epoch
        print('数据样例')
        print(data.head())
        gkf = StratifiedKFold(n_splits=kfold).split(X=data['input_ids'].fillna(0),
                                                    y=np.array(list(range(len(data)))) // 100)

        for fold, (train_idx, valid_idx) in enumerate(gkf):
            cudnn.benchmark = True
            print('=' * 40, 'fold: {}'.format(fold), '=' * 40)

            src = torch.LongTensor(data.iloc[train_idx]['input_ids'].values.tolist())
            seg = torch.LongTensor(data.iloc[train_idx]['segment_ids'].values.tolist())
            mask = torch.LongTensor(data.iloc[train_idx]['input_mask'].values.tolist())
            tgt = torch.LongTensor(data.iloc[train_idx]['label'].values.tolist())

            src_v = torch.LongTensor(data.iloc[valid_idx]['input_ids'].values.tolist())
            seg_v = torch.LongTensor(data.iloc[valid_idx]['segment_ids'].values.tolist())
            mask_v = torch.LongTensor(data.iloc[valid_idx]['input_mask'].values.tolist())
            tgt_v = torch.LongTensor(data.iloc[valid_idx]['label'].values.tolist())

            # 释放显存占用
            torch.cuda.empty_cache()

            train_num = len(src)
            train_steps = int(train_num * epoch / batch_size) + 1

            model = build_model(ptm_name, num_labels, type=model_type)
            optimizer, scheduler = build_optimizer(model, train_steps, learning_rate=learning_rate,
                                                   weight_decay=weight_decay, warmup_ratio=warmup_ratio)
            model.to(device)

            total_loss = 0.0
            global_steps = 0
            global_acc = 0
            bar = tqdm(range(1, epoch + 1))
            for e in bar:

                for i, (src_batch, tgt_batch, seg_batch, mask_batch) \
                        in enumerate(batch_loader(batch_size, src, tgt, seg, mask)):
                    src_batch = src_batch.to(device)
                    tgt_batch = tgt_batch.to(device)
                    seg_batch = seg_batch.to(device)
                    mask_batch = mask_batch.to(device)

                    output = model(input_ids=src_batch, labels=tgt_batch,
                                   token_type_ids=seg_batch, attention_mask=mask_batch)
                    loss = output[0]

                    optimizer.zero_grad()
                    loss.backward()

                    total_loss += loss.item()
                    # cur_avg_loss += loss.item()


                    scheduler.step()

                    global_steps += 1

                    acc = torch.argmax(torch.softmax(output[1], dim=-1), dim=-1) == tgt_batch
                    if model_type == 'cls':
                        acc = acc.type(torch.float)
                        acc = torch.mean(acc)
                    else:
                        acc = acc.type(torch.float)
                        acc = torch.sum(acc * mask_batch)
                        acc = acc / torch.sum(mask_batch)
                    bar.set_description('step:{} acc:{} loss:{} lr:{}'.format(
                        global_steps, round(acc.item(), 4), round(loss.item(), 4), round(scheduler.get_lr()[0], 7)))

                # 每个epoch结束进行验证集评估并保存模型
                labels = None
                outputs = None
                for k, (src_batch, tgt_batch, seg_batch, mask_batch) \
                        in enumerate(batch_loader(batch_size, src_v, tgt_v, seg_v, mask_v)):

                    src_batch = src_batch.to(device)
                    tgt_batch = tgt_batch.to(device)
                    seg_batch = seg_batch.to(device)
                    mask_batch = mask_batch.to(device)

                    output = model(input_ids=src_batch, labels=tgt_batch,
                                   token_type_ids=seg_batch, attention_mask=mask_batch)

                    output = output[1].cpu().detach().numpy()
                    tgt_batch = tgt_batch.cpu().detach().numpy()

                    output = np.argmax(output, axis=-1)
                    if labels is None:
                        labels = tgt_batch
                        outputs = output
                    else:
                        labels = np.concatenate([labels, tgt_batch], axis=0)
                        outputs = np.concatenate([outputs, output], axis=0)

                labels = np.reshape(labels, [-1])
                outputs = np.reshape(outputs, [-1])

                if model_type == 'tag':
                    used_index = labels > 0
                    labels = labels[used_index]
                    outputs = outputs[used_index]

                met = multi_cls_metrics(labels, outputs, need_sparse=False, num_labels=num_labels)
                acc = met['acc']

                table = PrettyTable(['global_steps',
                                     'loss',
                                     'lr',
                                     'acc'])
                table.add_row([global_steps + 1,
                               total_loss / (global_steps + 1),
                               scheduler.get_lr()[0],
                               round(acc, 4)])
                print(table)

                if global_acc < acc:
                    output_path_ = output_path + f'_fold{fold}'
                    if not os.path.exists(output_path_):
                        os.mkdir(output_path_)
                    model_save_path = os.path.join(output_path_, 'finetune_model')
                    model_to_save = model.module if hasattr(model, 'module') else model

                    model_to_save.save_pretrained(model_save_path)


if __name__ == '__main__':
    ft = FinetuneTrainer(epoch=100, kfold=5)
    print(ft.epoch)
    print(ft.kfold)
