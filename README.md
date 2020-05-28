# Source code of Boundary-aware Model for Nested NER
Implementation of Our Paper "A Boundary-aware Model for Nested Named Entity Recognition" in EMNLP-IJCNLP 2019.

# Requirements
* `python 3`
* `pytorch`
* `numpy`
* `gensim`
* `scikit-learn`
* `joblib`

# Data Format
Our processed `GENIA` dataset is in `./data/genia`.

The data format is the same as in [Neural Layered Model, Ju et al. 2018 NAACL](https://github.com/meizhiju/layered-bilstm-crf) 
>Each line has multiple columns separated by a tab key. 
>Each line contains
>```
>word	label1	label2	label3	...	labelN
>```
>The number of labels (`N`) for each word is determined by the maximum nested level in the data set. `N=maximum nested level + 1`
>Each sentence is separated by an empty line.
>For example, for these two sentences, `John killed Mary's husband. He was arrested last night` , they contain four entities: John (`PER`), Mary(`PER`), Mary's husband(`PER`),He (`PER`).
>The format for these two sentences is listed as following:
>```
>John    B-PER   O   O
>killed  O   O   O
>Mary    B-PER   B-PER   O
>'s  O   I-PER   O
>husband O   I-PER   O
>.   O   O   O
>
>He    B-PER   O   O
>was  O   O   O
>arrested  O   O   O
>last  O   O   O
>night  O   O   O
>.  O   O   O
>```

# Pre-trained word embeddings
## For GENIA Corpus:
* [Pre-trained word embeddings](https://drive.google.com/open?id=0BzMCqpcgEJgiUWs0ZnU0NlFTam8) used here is the same as in [Neural Layered Model](https://github.com/meizhiju/layered-bilstm-crf) 
## For GermEval 2014 Corpus: 
* [Pre-trained word embeddings](https://www.informatik.tudarmstadt.de/ukp/research_6/ukp_in_challenges/germeval_2014/index.en.jsp) used here is the same as in paper [GermEval-2014: Nested Named Entity Recognition with Neural Networks.](https://pdfs.semanticscholar.org/9b64/4bf5262e0d02d7ac25dab509d07d240b263a.pdf)

# Setup
Download pre-trained embedding above, unzip it, and place `PubMed-shuffle-win-30.bin` into `./data/embedding/`
For GermEval 2014 dataset, the same as GENIA.

# Usage
## Training

```sh
python3 train.py
```
trained model will saved at `./data/model/`
## Testing
 set `model_url` to the url of saved model in training in `main()` of `eval.py`
```sh
python3 eval.py
```

If you find this repo helpful, please cite the following:
```latex
@inproceedings{zheng2019boundary,
  title={A Boundary-aware Neural Model for Nested Named Entity Recognition},
  author={Zheng, Changmeng and Cai, Yi and Xu, Jingyun and Leung, Ho-fung and Xu, Guandong},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={357--366},
  year={2019}
}
```latex
