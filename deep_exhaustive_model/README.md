# An Implementation of Deep Exhaustive Model for Nested NER

Original paper: [Deep Exhaustive model, Soharb and Miwa (2018 EMNLP)](http://aclweb.org/anthology/D18-1309)

# Requirements
* `python 3`
* `ptorch`
* `numpy`
* `gensim`
* `scikit-learn`
* `joblib`

# Data Format
Our processed `GENIA` dataset is in `./data/`.

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
* [Pre-trained word embeddings](https://drive.google.com/open?id=0BzMCqpcgEJgiUWs0ZnU0NlFTam8) used here is the same as in [Neural Layered Model](https://github.com/meizhiju/layered-bilstm-crf) 

# Setup
Download pre-trained embedding above, unzip it, and place `PubMed-shuffle-win-30.bin` into `./data/embedding/`

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
