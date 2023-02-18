# coding: utf-8
from itertools import islice

import numpy as np
import orjson
from loguru import logger
from more_itertools import chunked
from sentence_transformers import SentenceTransformer

from constants import Directories

BATCH_SIZE = 3

model_names = [
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli',
    'DMetaSoul/sbert-chinese-general-v2',
    'DMetaSoul/sbert-chinese-general-v2-distill',
]


def inference_file(input_file, output_file, model: SentenceTransformer):
    with open(input_file, encoding='utf_8') as fin, \
            open(output_file, mode='w') as fout:
        fin = islice(fin, 10)
        for chunk in chunked(fin, BATCH_SIZE):
            lines = [orjson.loads(line) for line in chunk]
            sentence1_list = [j['sentence1'] for j in lines]
            sentence2_list = [j['sentence2'] for j in lines]
            labels = [j['label'] for j in lines]
            embeddings1 = model.encode(sentence1_list, normalize_embeddings=True)
            embeddings2 = model.encode(sentence2_list, normalize_embeddings=True)
            cos_sims = np.sum(embeddings1 * embeddings2, axis=1)
            for cos_sim, label in zip(cos_sims, labels):
                fout.write(f'{label},{cos_sim:.3f}\n')


for model_name in model_names:
    logger.info(f'----- {model_name} -----')
    model = SentenceTransformer(model_name)

    # train
    input_file = Directories.DATA / 'train.json'
    output_file = Directories.DATA / f'train-result-{model_name.replace("/", "_")}.txt'
    inference_file(input_file, output_file, model)

    # dev
    input_file = Directories.DATA / 'dev.json'
    output_file = Directories.DATA / f'dev-result-{model_name.replace("/", "_")}.txt'
    inference_file(input_file, output_file, model)

    # test
    input_file = Directories.DATA / 'test.json'
    output_file = Directories.DATA / f'test-result-{model_name.replace("/", "_")}.txt'
    inference_file(input_file, output_file, model)
