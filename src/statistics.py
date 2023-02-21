# coding: utf-8
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from constants import Directories

model_names = [
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli',
    'DMetaSoul/sbert-chinese-general-v2',
    'DMetaSoul/sbert-chinese-general-v2-distill',
]


def get_f1(file):
    labels = []
    preds = []
    with open(file, encoding='utf_8') as fin:
        for line in fin:
            label, pred = line.strip().split(',')
            labels.append(int(label))
            preds.append(float(pred))
        labels = np.asarray(labels)
        preds = np.asarray(preds)
        return f1_score(labels, np.where(preds >= 0.5, 1, 0)), roc_auc_score(labels, preds)


def get_speed(file):
    with open(file, encoding='utf_8') as fin:
        return float(fin.read().strip())


for model_name in model_names:
    stats = {}
    model_name_r = '_'.join(model_name.split('/'))
    for task in ('dev', 'test'):
        f1, auc = get_f1(Directories.DATA_RESULT / f'{task}-result-{model_name_r}.txt')
        speed = get_speed(Directories.DATA_RESULT / f'{task}-speed-{model_name_r}.txt')
        stats[task] = f'{f1 * 100:.1f} / {auc * 100:.1f} / {speed:.2f}'
    print(f'|{model_name}|{stats["dev"]}|{stats["test"]}|')
