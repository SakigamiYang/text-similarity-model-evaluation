# text-similarity-model-evaluation

Evaluation of several non-personally developed pre-trained text semantic similarity models on SimCLUE ( a Chinese text
semantic similarity test set ) .

## Dataset

SimCLUE: https://github.com/CLUEbenchmark/SimCLUE

List of integrated datasets:

- LCQMC
- AFQMC
- OPPO Xiaobu
- PKU-Paraphrase-Bank
- Chinese-STS-B
- Chinese-MNLI
- Chinese-SNLI
- OCNLI
- CINLID

## Result

| model_name                                                  | f1/auc/speed (test)      | f1/auc/speed (dev)       |
|-------------------------------------------------------------|--------------------------|--------------------------|
| sentence-transformers/distiluse-base-multilingual-cased-v2  | 64.7 / 71.4 / 163.85     | 66.6 / 74.8 / 163.70     |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 68.2 / 76.6 / **345.99** | 68.7 / 78.9 / **407.74** |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 67.9 / 76.9 / 145.30     | 68.3 / 79.2 / 145.78     |
| symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli             | **70.0** / 78.0 / 148.80 | **70.7** / 79.6 / 145.59 |
| DMetaSoul/sbert-chinese-general-v2                          | 62.6 / **92.3** / 110.68 | 63.7 / **92.7** / 111.24 |
| DMetaSoul/sbert-chinese-general-v2-distill                  | 61.5 / 82.0 / 313.89     | 63.2 / 84.1 / 300.35     |
