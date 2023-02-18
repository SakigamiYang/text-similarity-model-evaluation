# coding: utf-8
import orjson
from loguru import logger

from constants import Directories

# train
logger.info(f'start make train.json')

train_file = Directories.DATA / 'train.json'

if train_file.exists():
    train_file.unlink()

with open(Directories.DATA_SIMCLUE / 'train_pair.json', encoding='utf_8') as fin, \
        open(train_file, mode='wb+') as fout:
    for line in fin:
        line = line.rstrip()
        if line:
            j = orjson.loads(line)
            j['label'] = int(j['label'])
            fout.write(orjson.dumps(j))
            fout.write(b'\n')

with open(Directories.DATA_SIMCLUE / 'train_rank.json', encoding='utf_8') as fin, \
        open(train_file, mode='ab+') as fout:
    for line in fin:
        line = line.rstrip()
        if line:
            j = orjson.loads(line)
            fout.write(orjson.dumps({'sentence1': j['query'], 'sentence2': j['title'], 'label': 1}))
            fout.write(b'\n')
            fout.write(orjson.dumps({'sentence1': j['query'], 'sentence2': j['neg_title'], 'label': 0}))
            fout.write(b'\n')

logger.info(f'finish make train.json')

# dev
logger.info(f'start make dev.json')

dev_file = Directories.DATA / 'dev.json'

if dev_file.exists():
    dev_file.unlink()

with open(Directories.DATA_SIMCLUE / 'dev.json', encoding='utf_8') as fin, \
        open(dev_file, mode='wb') as fout:
    for line in fin:
        line = line.rstrip()
        if line:
            j = orjson.loads(line)
            j['label'] = int(j['label'])
            fout.write(orjson.dumps(j))
            fout.write(b'\n')

logger.info(f'finish make dev.json')

# test
logger.info(f'start make test.json')

test_file = Directories.DATA / 'test.json'

if test_file.exists():
    test_file.unlink()

with open(Directories.DATA_SIMCLUE / 'test_public.json', encoding='utf_8') as fin, \
        open(test_file, mode='wb') as fout:
    for line in fin:
        line = line.rstrip()
        if line:
            j = orjson.loads(line)
            j['label'] = int(j['label'])
            fout.write(orjson.dumps(j))
            fout.write(b'\n')

logger.info(f'finish make test.json')
