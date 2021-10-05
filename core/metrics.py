# coding:utf8

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


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
