# coding:utf-8


import os
import json
import time
import warnings
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.backends import cudnn

from transformers import (
    BertTokenizer,
    BertTokenizerFast,
)

from core.modules import BertForSequenceClassification
from core.optimizer import WarmupLinearSchedule
from core.utils import batch_loader, read_dataset


def build_model_and_tokenizer(config):
    tokenizer_path = config['model_path'] + '/vocab.txt'
    if config['tokenizer_fast']:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    model = BertForSequenceClassification.from_pretrained(config['model_path'], num_labels=config['num_labels'])

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


def main():
    config = {
        'num_labels': 3,
        'use_model': 'bert',
        'normal_data_cache_path': 'user_data/processed/bert/all_data.pkl',  # 96423
        'train_data_path': 'data/data.csv',
        'dev_data_path': 'data/test.csv',
        'output_path': 'output_model_cls',
        # 'model_path': '/chj/dev/lianxiaolei/model/bert-base-chinese',  # your pretrain model path
        'model_path': '/chj/dev/lianxiaolei/bert_pretrain/mlm_model_prod_opin',  # your pretrain model path
        'shuffle_way': '',
        'use_swa': True,
        'tokenizer_fast': False,
        'batch_size': 128,
        'num_epochs': 20,
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
        'seed': 2021,
    }

    warnings.filterwarnings('ignore')
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))
    print("\n>> loading model from :{}".format(config['model_path']))

    tokenizer, model = build_model_and_tokenizer(config)

    dataset = read_dataset(config['train_data_path'], tokenizer, max_seq_len=config['max_seq_len'])

    train_num = len(dataset)
    train_steps = int(train_num * config['num_epochs'] / config['batch_size']) + 1

    optimizer, scheduler = build_optimizer(config, model, train_steps)
    model.to(config['device'])

    src = torch.LongTensor(dataset['input_ids'])
    seg = torch.LongTensor(dataset['segment_ids'])
    mask = torch.LongTensor(dataset['input_mask'])
    tgt = torch.LongTensor(dataset['label'])

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
                           token_type_ids=seg_batch, attention_mask=mask_batch, label_balance=False)
            loss = output[0]
            optimizer.zero_grad()
            loss.backward()

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            optimizer.step()

            scheduler.step()
            # model.zero_grad()

            # acc = torch.argmax(torch.softmax(output[1], dim=-1), dim=-1) == tgt_batch
            # print('data:\n', torch.argmax(torch.softmax(output[1], dim=-1), dim=-1), '\n', tgt_batch)
            acc = torch.argmax(torch.softmax(output[1], dim=-1), dim=-1) == tgt_batch
            acc = acc.type(torch.float)
            acc = torch.mean(acc)
            bar.set_description('step:{} acc:{} loss:{} lr:{}'.format(
                global_steps, round(acc.item(), 4), round(loss.item(), 4), round(scheduler.get_lr()[0], 7)))

            # if (global_steps + 1) % config['logging_step'] == 0:
            #     labels = None
            #     outputs = None
            #     for k, (src_batch, tgt_batch, seg_batch, mask_batch) \
            #             in enumerate(batch_loader(config, src_t, tgt_t, seg_t, mask_t)):
            #
            #         src_batch = src_batch.to(config['device'])
            #         tgt_batch = tgt_batch.to(config['device'])
            #         seg_batch = seg_batch.to(config['device'])
            #         mask_batch = mask_batch.to(config['device'])
            #
            #         output = model(input_ids=src_batch, labels=tgt_batch,
            #                        token_type_ids=seg_batch, attention_mask=mask_batch)
            #
            #         output = output[1].cpu().detach().numpy()
            #         tgt_batch = tgt_batch.cpu().detach().numpy()
            #
            #         output = np.argmax(output, axis=-1)
            #         if labels is None:
            #             labels = tgt_batch
            #             outputs = output
            #         else:
            #             labels = np.concatenate([labels, tgt_batch], axis=0)
            #             outputs = np.concatenate([outputs, output], axis=0)
            #
            #     labels = np.reshape(labels, [-1])
            #     outputs = np.reshape(outputs, [-1])
            #
            #     # used_index = labels > 0
            #     # labels = labels[used_index]
            #     # outputs = outputs[used_index]
            #
            #     met = multi_cls_metrics(labels, outputs, need_sparse=False, num_labels=config['num_labels'])
            #     acc = met['acc']
            #
            #     table = PrettyTable(['global_steps',
            #                          'loss',
            #                          'lr',
            #                          'acc',
            #                          'time_cost'])
            #     table.add_row([global_steps + 1,
            #                    total_loss / (global_steps + 1),
            #                    scheduler.get_lr()[0],
            #                    round(acc, 4),
            #                    round((time.time() - start) / 60.00, 1)])
            #     print(table)
            #
            #     cur_avg_loss = 0.0
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


if __name__ == '__main__':
    main()
