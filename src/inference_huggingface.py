# coding: utf-8
import os
from time import perf_counter

import numpy as np
import orjson
from dotenv import load_dotenv
from loguru import logger
from more_itertools import chunked
from sentence_transformers import SentenceTransformer

from constants import Directories

load_dotenv()

BATCH_SIZE = 32

model_names = [
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli',
    'DMetaSoul/sbert-chinese-general-v2',
    'DMetaSoul/sbert-chinese-general-v2-distill',
]


def inference_file(input_file, output_file, model: SentenceTransformer):
    total_samples = 0
    total_time = 0.

    with open(input_file, encoding='utf_8') as fin, \
            open(output_file, mode='w') as fout:
        for chunk in chunked(fin, BATCH_SIZE):
            total_samples += len(chunk)

            lines = [orjson.loads(line) for line in chunk]
            sentence1_list = [j['sentence1'] for j in lines]
            sentence2_list = [j['sentence2'] for j in lines]
            labels = [int(j['label']) for j in lines]

            start = perf_counter()

            embeddings1 = model.encode(sentence1_list, normalize_embeddings=True)
            embeddings2 = model.encode(sentence2_list, normalize_embeddings=True)
            cos_sims = np.sum(embeddings1 * embeddings2, axis=1)

            end = perf_counter()
            total_time += end - start

            for cos_sim, label in zip(cos_sims, labels):
                fout.write(f'{label},{cos_sim:.3f}\n')

    logger.info(total_samples / total_time)


for model_name in model_names:
    logger.info(f'----- {model_name} -----')
    model = SentenceTransformer(model_name, cache_folder=os.environ.get('SENTENCE_TRANSFORMERS_HOME'))

    # dev
    logger.info(f'start dev result')
    input_file = Directories.DATA_SIMCLUE / 'dev.json'
    output_file = Directories.DATA_RESULT / f'dev-result-{model_name.replace("/", "_")}.txt'
    inference_file(input_file, output_file, model)
    logger.info(f'finish dev result')

    # test
    logger.info(f'start test result')
    input_file = Directories.DATA_SIMCLUE / 'test_public.json'
    output_file = Directories.DATA_RESULT / f'test-result-{model_name.replace("/", "_")}.txt'
    inference_file(input_file, output_file, model)
    logger.info(f'finish test result')
