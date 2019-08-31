# coding: utf-8

import os
import numpy as np
import joblib
import torch
from collections import defaultdict
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.path_util import from_project_root, dirname
import utils.json_util as ju

np.random.seed(233)
LABEL_IDS = {"DNA": 1, "RNA": 2, "protein": 3, "cell_line": 4, "cell_type": 5}
PRETRAINED_URL = from_project_root("data/embedding/PubMed-shuffle-win-30.bin")


def gen_sentence_tensors(sentence_list, device, data_url):
    """ generate input tensors from sentence list

    Args:
        sentence_list: list of raw sentence
        device: torch device
        data_url: raw data url to locate the vocab url

    Returns:
        sentences, tensor
        sentence_lengths, tensor
        sentence_words, list of tensor
        sentence_word_lengths, list of tensor
        sentence_word_indices, list of tensor

    """
    vocab = ju.load(dirname(data_url) + '/vocab.json')
    char_vocab = ju.load(dirname(data_url) + '/char_vocab.json')

    sentences = list()
    sentence_words = list()
    sentence_word_lengths = list()
    sentence_word_indices = list()

    unk_idx = 1
    for sent in sentence_list:
        # word to word id
        sentence = torch.LongTensor([vocab[word] if word in vocab else unk_idx
                                     for word in sent]).to(device)

        # char of word to char id
        words = list()
        for word in sent:
            words.append([char_vocab[ch] if ch in char_vocab else unk_idx
                          for ch in word])

        # save word lengths
        word_lengths = torch.LongTensor([len(word) for word in words]).to(device)

        # sorting lengths according to length
        word_lengths, word_indices = torch.sort(word_lengths, descending=True)

        # sorting word according word length
        words = np.array(words)[word_indices.cpu().numpy()]
        word_indices = word_indices.to(device)
        words = [torch.LongTensor(word).to(device) for word in words]

        # padding char tensor of words
        words = pad_sequence(words, batch_first=True).to(device)
        # (max_word_len, sent_len)

        sentences.append(sentence)
        sentence_words.append(words)
        sentence_word_lengths.append(word_lengths)
        sentence_word_indices.append(word_indices)

    # record sentence length and padding sentences
    sentence_lengths = [len(sentence) for sentence in sentences]
    # (batch_size)
    sentences = pad_sequence(sentences, batch_first=True).to(device)
    # (batch_size, max_sent_len)

    return sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices


class ExhaustiveDataset(Dataset):
    label_ids = {"neither": 0, "DNA": 1, "RNA": 2, "protein": 3, "cell_line": 4, "cell_type": 5, "padding": 6}

    def __init__(self, data_url, device, max_region=10):
        super().__init__()
        self.x, self.y = load_raw_data(data_url)
        self.data_url = data_url
        self.max_region = max_region
        self.device = device

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def collate_func(self, data_list):
        data_list = sorted(data_list, key=lambda tup: len(tup[0]), reverse=True)
        sentence_list, records_list = zip(*data_list)  # un zip
        max_sent_len = len(sentence_list[0])
        sentence_tensors = gen_sentence_tensors(sentence_list, self.device, self.data_url)
        # (sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices)

        region_labels = list()
        for records, length in zip(records_list, sentence_tensors[1]):
            labels = list()
            for region_size in range(1, self.max_region + 1):
                for start in range(0, max_sent_len - region_size + 1):
                    if start + region_size > length:
                        labels.append(6)
                    elif (start, start + region_size) in records:
                        labels.append(self.label_ids[records[start, start + region_size]])
                    else:
                        labels.append(0)
            region_labels.append(labels)
        region_labels = torch.LongTensor(region_labels).to(self.device)
        # (batch_size, n_regions)

        return sentence_tensors, region_labels, records_list


def gen_vocab_from_data(data_urls, pretrained_url, binary=True, update=False, min_count=1):
    """ generate vocabulary and embeddings from data file, generated vocab files will be saved in
        data dir

    Args:
        data_urls: url to data file(s), list or string
        pretrained_url: url to pretrained embedding file
        binary: binary for load word2vec
        update: force to update even vocab file exists
        min_count: minimum count of a word

    Returns:
        generated word embedding url
    """

    if isinstance(data_urls, str):
        data_urls = [data_urls]
    data_dir = os.path.dirname(data_urls[0])
    vocab_url = os.path.join(data_dir, "vocab.json")
    char_vocab_url = os.path.join(data_dir, "char_vocab.json")
    embedding_url = os.path.join(data_dir, "embeddings.npy") if pretrained_url else None

    if (not update) and os.path.exists(vocab_url):
        print("vocab file already exists")
        return embedding_url

    vocab = set()
    char_vocab = set()
    word_counts = defaultdict(int)
    print("generating vocab from", data_urls)
    for data_url in data_urls:
        with open(data_url, 'r', encoding='utf-8') as data_file:
            for row in data_file:
                if row == '\n':
                    continue
                token = row.split()[0]
                word_counts[token] += 1
                if word_counts[token] > min_count:
                    vocab.add(row.split()[0])
                char_vocab = char_vocab.union(row.split()[0])

    # sorting vocab according alphabet order
    vocab = sorted(vocab)
    char_vocab = sorted(char_vocab)

    # generate word embeddings for vocab
    if pretrained_url is not None:
        print("generating pre-trained embedding from", pretrained_url)
        kvs = KeyedVectors.load_word2vec_format(pretrained_url, binary=binary)
        embeddings = list()
        for word in vocab:
            if word in kvs:
                embeddings.append(kvs[word])
            else:
                embeddings.append(np.random.uniform(-0.25, 0.25, kvs.vector_size)),

    char_vocab = ['<pad', '<unk>'] + char_vocab
    vocab = ['<pad>', '<unk>'] + vocab
    ju.dump(ju.list_to_dict(vocab), vocab_url)
    ju.dump(ju.list_to_dict(char_vocab), char_vocab_url)

    if pretrained_url is None:
        return
    embeddings = np.vstack([np.zeros(kvs.vector_size),  # for <pad>
                            np.random.uniform(-0.25, 0.25, kvs.vector_size),  # for <unk>
                            embeddings])
    np.save(embedding_url, embeddings)
    return embedding_url


def infer_records(columns):
    """ inferring all entity records of a sentence
    Args:
        columns: columns of a sentence in iob2 format
    Returns:
        entity record in gave sentence
    """
    records = dict()
    for col in columns:
        start = 0
        while start < len(col):
            end = start + 1
            if col[start][0] == 'B':
                while end < len(col) and col[end][0] == 'I':
                    end += 1
                records[(start, end)] = col[start][2:]
            start = end
    return records


def load_raw_data(data_url, update=False):
    """ load data into sentences and records

    Args:
        data_url: url to data file
        update: whether force to update
    Returns:
        sentences(raw), records
    """

    # load from pickle
    save_url = data_url.replace('.bio', '.raw.pkl').replace('.iob2', '.raw.pkl')
    if not update and os.path.exists(save_url):
        return joblib.load(save_url)

    sentences = list()
    records = list()
    with open(data_url, 'r', encoding='utf-8') as iob_file:
        first_line = iob_file.readline()
        n_columns = first_line.count('\t')
        # JNLPBA dataset don't contains the extra 'O' column
        if 'jnlpba' in data_url:
            n_columns += 1
        columns = [[x] for x in first_line.split()]
        for line in iob_file:
            if line != '\n':
                line_values = line.split()
                for i in range(n_columns):
                    columns[i].append(line_values[i])

            else:  # end of a sentence
                sentence = columns[0]
                sentences.append(sentence)
                records.append(infer_records(columns[1:]))
                columns = [list() for i in range(n_columns)]
    joblib.dump((sentences, records), save_url)
    return sentences, records


def prepare_vocab(data_urls, pretrained_url=PRETRAINED_URL, update=True, min_count=1):
    """ prepare vocab and embedding

    Args:
        data_urls: urls to data file for preparing vocab
        pretrained_url: url to pretrained embedding file
        min_count: minimum count of word
        update: force to update

    """
    binary = pretrained_url.endswith('.bin')
    gen_vocab_from_data(data_urls, pretrained_url, binary=binary, update=update, min_count=min_count)


def main():
    data_urls = [from_project_root("data/genia.train.iob2"),
                 from_project_root("data/genia.dev.iob2"),
                 from_project_root("data/genia.test.iob2")]
    prepare_vocab(data_urls, update=True, min_count=1)
    pass


if __name__ == '__main__':
    main()
