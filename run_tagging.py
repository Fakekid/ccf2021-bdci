# coding:utf-8

"""
cls和tag联合任务，取cls位的pred为cls结果，其它位置的为tag结果
"""

import os
import json
import time
import pickle
import random
import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
# from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.backends import cudnn
from torch.optim.lr_scheduler import LambdaLR
# from transformers.modeling_bert import SequenceClassifierOutput
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score

from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    BertModel,
    BertPreTrainedModel,
    BertForTokenClassification
)
from transformers.modeling_outputs import TokenClassifierOutput

from core.modules import LabelSmoothingLoss


class BertForTokenClassification_(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 转移矩阵的参数初始化，transitions[i,j]代表的是从第j个tag转移到第i个tag的转移分数
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = LabelSmoothingLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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


def read_dataset(data_path, tokenizer, max_seq_len=128):
    start = time.time()
    dataset = []
    seq_length = max_seq_len

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(tqdm(f)):
            if line_id == 0: continue
            sent, tag, tgt = line.strip().split('\t')
            src = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(sent) + ['SEP'])
            tag = tag.split(' ')
            tgt = [int(tgt) + 10] + list(map(int, tag)) + [int(1)]

            assert len(src) == len(tgt), \
                ValueError('行{}文本序列和标注序列长度不一致：src length:{}, target length:{}\n sequence content:{}\n{}\n{}'.format(
                    line_id, len(src), len(tgt), src, tgt, sent))

            seg = [0] * len(src)
            mask = [1] * len(src)

            if len(src) > seq_length:
                src = src[: seq_length]
                tgt = tgt[: seq_length]
                seg = seg[: seq_length]
                mask = mask[: seq_length]
            while len(src) < seq_length:
                tgt.append(0)
                src.append(0)
                seg.append(0)
                mask.append(0)
            dataset.append((src, tgt, seg, mask))

    return dataset


def read_dataset_for_tag(data_path, tokenizer, max_seq_len=128):
    start = time.time()
    dataset = []
    seq_length = max_seq_len

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(tqdm(f)):
            if line_id == 0: continue
            sent, tag, tgt = line.strip().split('\t')
            src = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(sent) + ['SEP'])
            tag = tag.split(' ')
            tgt = [1] + list(map(int, tag)) + [1]

            assert len(src) == len(tgt), \
                ValueError('行{}文本序列和标注序列长度不一致：src length:{}, target length:{}\n sequence content:{}\n{}\n{}'.format(
                    line_id, len(src), len(tgt), src, tgt, sent))

            seg = [0] * len(src)
            mask = [1] * len(src)

            if len(src) > seq_length:
                src = src[: seq_length]
                tgt = tgt[: seq_length]
                seg = seg[: seq_length]
                mask = mask[: seq_length]
            while len(src) < seq_length:
                tgt.append(0)
                src.append(0)
                seg.append(0)
                mask.append(0)
            dataset.append((src, tgt, seg, mask))

    return dataset


class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.01):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FGM:
    def __init__(self, config, model):
        self.model = model
        self.backup = {}
        self.emb_name = config['emb_name']
        self.epsilon = config['epsilon']

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, config, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = config['epsilon']
        self.emb_name = config['emb_name']
        self.alpha = config['alpha']

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


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


def build_model_and_tokenizer(config):
    tokenizer_path = config['model_path'] + '/vocab.txt'
    if config['tokenizer_fast']:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    model = BertForTokenClassification.from_pretrained(config['model_path'], num_labels=config['num_labels'])

    return tokenizer, model


def build_optimizer(config, model, train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': config['weight_decay']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * config['warmup_ratio'],
                                     t_total=train_steps)

    return optimizer, scheduler


def log_sum_exp(vec):
    def argmax():
        # return the argmax as a python int
        # 返回vec的dim为1维度上的最大值索引
        _, idx = torch.max(vec, 1)
        return idx.item()

    max_score = vec[0, argmax()]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def main():
    config = {
        'num_labels': 10,
        'use_model': 'bert',
        'normal_data_cache_path': 'user_data/processed/bert/all_data.pkl',  # 96423
        'train_data_path': 'data/train.csv',
        'dev_data_path': 'data/test.csv',
        'output_path': 'output_model_tag',
        # 'model_path': '/chj/dev/lianxiaolei/model/bert-base-chinese',  # your pretrain model path
        'model_path': '/chj/dev/lianxiaolei/bert_pretrain/mlm_model_prod_opin',  # your pretrain model path
        'shuffle_way': '',
        'use_swa': True,
        'tokenizer_fast': False,
        'batch_size': 64,
        'num_epochs': 30,
        'max_seq_len': 256,
        'learning_rate': 2e-5,
        'alpha': 0.3,
        'epsilon': 1.0,
        'adv_k': 3,
        'emb_name': 'word_embeddings.',
        'adv': 'pgd',
        'warmup_ratio': 0.05,
        'weight_decay': 0.01,
        'device': 'cuda',
        'logging_step': 100,
        'seed': 2021
    }

    warnings.filterwarnings('ignore')
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))
    print("\n>> loading model from :{}".format(config['model_path']))

    tokenizer, model = build_model_and_tokenizer(config)
    if not os.path.exists(config['normal_data_cache_path']):
        train_set = read_dataset_for_tag(config['train_data_path'], tokenizer, max_seq_len=config['max_seq_len'])
        dev_set = read_dataset_for_tag(config['dev_data_path'], tokenizer, max_seq_len=config['max_seq_len'])
    else:
        with open(config['normal_data_cache_path'], 'rb') as f:
            train_set = pickle.load(f)

    seed_everything(config['seed'])

    if config['shuffle_way'] == 'block_shuffle':
        train_set = block_shuffle(config, train_set)
    else:
        random.shuffle(train_set)

    train_num = len(train_set)
    train_steps = int(train_num * config['num_epochs'] / config['batch_size']) + 1

    optimizer, scheduler = build_optimizer(config, model, train_steps)
    model.to(config['device'])

    src = torch.LongTensor([example[0] for example in train_set])
    tgt = torch.LongTensor([example[1] for example in train_set])
    seg = torch.LongTensor([example[2] for example in train_set])
    mask = torch.LongTensor([example[3] for example in train_set])

    src_t = torch.LongTensor([example[0] for example in dev_set])
    tgt_t = torch.LongTensor([example[1] for example in dev_set])
    seg_t = torch.LongTensor([example[2] for example in dev_set])
    mask_t = torch.LongTensor([example[3] for example in dev_set])

    cudnn.benchmark = True

    total_loss, cur_avg_loss = 0.0, 0.0
    global_steps = 0

    if config['adv'] == '':
        print('\n>> start normal training ...')
    elif config['adv'] == 'fgm':
        print('\n>> start fgm training ...')
    elif config['adv'] == 'pgd':
        print('\n>> start pgd training ...')

    start = time.time()
    bar = tqdm(range(1, config['num_epochs'] + 1))
    for epoch in bar:
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, mask_batch) \
                in enumerate(batch_loader(config, src, tgt, seg, mask)):

            src_batch = src_batch.to(config['device'])
            tgt_batch = tgt_batch.to(config['device'])
            seg_batch = seg_batch.to(config['device'])
            mask_batch = mask_batch.to(config['device'])

            output = model(input_ids=src_batch, labels=tgt_batch,
                           token_type_ids=seg_batch, attention_mask=mask_batch)
            loss = output[0]
            optimizer.zero_grad()
            loss.backward()

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            # if config['adv'] == 'fgm':
            #     fgm = FGM(config, model)
            #     fgm.attack()
            #     adv_loss = model(input_ids=src_batch, labels=tgt_batch,
            #                      token_type_ids=seg_batch, attention_mask=mask_batch)[0]
            #     adv_loss.backward()
            #     fgm.restore()
            #
            # if config['adv'] == 'pgd':
            #     pgd = PGD(config, model)
            #     pgd.backup_grad()
            #
            #     K = config['adv_k']
            #     for t in range(K):
            #         pgd.attack(is_first_attack=(t == 0))
            #         if t != K - 1:
            #             model.zero_grad()  # 在K次攻击中，pgd里已经记录下了梯度，故model的梯度可直接设为0
            #         else:
            #             pgd.restore_grad()  # 在K次攻击后，用pgd里的梯度重置model的梯度，完成K步梯度对抗
            #
            #         adv_loss = model(input_ids=src_batch, labels=tgt_batch,
            #                          token_type_ids=seg_batch, attention_mask=mask_batch)[0]
            #         adv_loss.backward()
            #     pgd.restore()

            optimizer.step()

            scheduler.step()
            # model.zero_grad()

            # acc = torch.argmax(torch.softmax(output[1], dim=-1), dim=-1) == tgt_batch
            # print('data:\n', torch.argmax(torch.softmax(output[1], dim=-1), dim=-1), '\n', tgt_batch)
            acc = torch.argmax(torch.softmax(output[1], dim=-1), dim=-1) == tgt_batch
            acc = acc.type(torch.float)
            acc = torch.sum(acc * mask_batch)
            acc = acc / torch.sum(mask_batch)
            bar.set_description('step:{} acc:{} loss:{} lr:{}'.format(
                global_steps, round(acc.item(), 4), round(loss.item(), 4), round(scheduler.get_lr()[0], 7)))

            if (global_steps + 1) % config['logging_step'] == 0:
                labels = None
                outputs = None
                for k, (src_batch, tgt_batch, seg_batch, mask_batch) \
                        in enumerate(batch_loader(config, src_t, tgt_t, seg_t, mask_t)):

                    src_batch = src_batch.to(config['device'])
                    tgt_batch = tgt_batch.to(config['device'])
                    seg_batch = seg_batch.to(config['device'])
                    mask_batch = mask_batch.to(config['device'])

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

                used_index = labels > 0
                labels = labels[used_index]
                outputs = outputs[used_index]

                met = multi_cls_metrics(labels, outputs, need_sparse=False, num_labels=config['num_labels'])
                acc = met['acc']

                table = PrettyTable(['global_steps',
                                     'loss',
                                     'lr',
                                     'acc',
                                     'time_cost'])
                table.add_row([global_steps + 1,
                               total_loss / (global_steps + 1),
                               scheduler.get_lr()[0],
                               round(acc, 4),
                               round((time.time() - start) / 60.00, 1)])
                print(table)

                cur_avg_loss = 0.0
            global_steps += 1

        model_save_path = os.path.join(config['output_path'], f'checkpoint-{global_steps}')
        model_to_save = model.module if hasattr(model, 'module') else model

        model_to_save.save_pretrained(model_save_path)

        conf = json.dumps(config)
        out_conf_path = os.path.join(config['output_path'], f'checkpoint-{global_steps}' +
                                     '/train_config.json')
        with open(out_conf_path, 'w', encoding='utf-8') as f:
            f.write(conf)

    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at:{}".format(localtime_end))


def multi_cls_metrics(labels, logits, average='micro', return_auc_when_multi_cls=None, need_sparse=True,
                      has_mask=False, num_labels=None):
    if num_labels is None:
        all_labels = None
    else:
        all_labels = [list(range(1, num_labels))]

    if need_sparse:
        preds = np.argmax(logits, axis=-1).reshape(-1)
    else:
        preds = logits.reshape(-1)

    labels = labels.reshape(-1)

    if has_mask:
        mask = labels > 0
        labels = labels[mask]
        preds = preds[mask]

    acc = np.mean(preds == labels)
    # acc = np.sum((preds == labels) * mask) / np.sum(mask)

    # f1 = f1_score(labels, preds, average=average, zero_division=0, labels=all_labels)
    #
    # p = precision_score(labels, preds, average=average, zero_division=0, labels=all_labels)
    # r = recall_score(labels, preds, average=average, zero_division=0, labels=all_labels)
    f1, p, r = 0, 0, 0

    if logits.shape[-1] > 2:
        if return_auc_when_multi_cls is not None:
            return {'auc': return_auc_when_multi_cls, 'acc': acc, 'f1': f1, 'p': p, 'r': r}
        else:
            return {'acc': acc, 'f1': f1, 'p': p, 'r': r}
    else:
        auc = roc_auc_score(labels, logits[:, 1], average=average)
    return {'auc': auc, 'acc': acc, 'f1': f1, 'p': p, 'r': r}


if __name__ == '__main__':
    main()
