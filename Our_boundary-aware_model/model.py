# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


class End2EndModel(nn.Module):

    def __init__(self, hidden_size, n_tags, embedding_url=None, bidirectional=True, lstm_layers=1,
                 n_embeddings=None, embedding_dim=None, freeze=False, char_feat_dim=50):
        super().__init__()

        if embedding_url:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(np.load(embedding_url)),
                freeze=freeze
            )
        else:
            self.embedding = nn.Embedding(n_embeddings, embedding_dim, padding_idx=0)

        self.embedding_dim = self.embedding.embedding_dim
        self.char_feat_dim = char_feat_dim
        self.word_repr_dim = self.embedding_dim + self.char_feat_dim

        self.char_repr = CharLSTM(
            n_chars=1000,
            embedding_size=char_feat_dim // 2,
            hidden_size=char_feat_dim // 2,
        ) if char_feat_dim > 0 else None

        self.dropout = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(
            input_size=self.word_repr_dim,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.lstm_layers = lstm_layers
        self.n_tags = n_tags
        self.n_hidden = (1 + bidirectional) * hidden_size


        self.region_clf = RegionCLF(
            input_dim=self.n_hidden,
            n_classes=n_tags,
        )

        # head and tail features classifier
        self.ht_labeler = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.n_hidden, 4),
        )

    def forward(self, sentences, sentence_lengths, sentence_words, sentence_word_lengths,
                sentence_word_indices, sentence_labels=None):

        # sentences (batch_size, max_sent_len)
        # sentence_length (batch_size)
        word_repr = self.embedding(sentences)
        # word_feat shape: (batch_size, max_sent_len, embedding_dim)

        # add character level feature
        if self.char_feat_dim > 0:
            # sentence_words (batch_size, *sent_len, max_word_len)
            # sentence_word_lengths (batch_size, *sent_len)
            # sentence_word_indices (batch_size, *sent_len, max_word_len)
            # char level feature
            char_feat = self.char_repr(sentence_words, sentence_word_lengths, sentence_word_indices)
            # char_feat shape: (batch_size, max_sent_len, char_feat_dim)

            # concatenate char level representation and word level one
            word_repr = torch.cat([word_repr, char_feat], dim=-1)
            # word_repr shape: (batch_size, max_sent_len, word_repr_dim)

        # drop out
        word_repr = self.dropout(word_repr)

        packed = nn.utils.rnn.pack_padded_sequence(word_repr, sentence_lengths, batch_first=True)
        out, (hn, _) = self.lstm(packed)
        # out packed_sequence(batch_size, max_sent_len, n_hidden)
        # hn (n_layers * n_directions, batch_size, hidden_size)

        max_sent_len = sentences.shape[1]
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=max_sent_len, batch_first=True)
        # unpacked (batch_size, max_sent_len, n_hidden)

        # self-attention for boosting boundary detection module
        # unpacked,attn_weight = attention(unpacked,unpacked,unpacked)

        # task1: head and tail sequence labeler
        sent_first = unpacked.transpose(0, 1)
        # sent_first (max_sent_len, batch_size, n_hidden)
        sentence_outputs = torch.stack([self.ht_labeler(token) for token in sent_first], dim=-1)
        # shape of each ht_labeler output: (batch_size, n_classes)
        # shape of sentence_outputs: (batch_size, n_classes, lengths[0])

        # task2: region classification
        # if not self.training:
        if sentence_labels is None:
            sentence_labels = torch.argmax(sentence_outputs, dim=1)
            # sentence_labels (batch_size, lengths[0])

        regions = list()
        for hidden, sentence_label, length in zip(unpacked, sentence_labels, sentence_lengths):
            for start in range(0, length):
                if sentence_label[start] == 1:
                    regions.append(hidden[start:start+1])
                    for end in range(start+1, length):
                        if sentence_label[end] == 2:
                            regions.append(hidden[start:end+1])

        # regions = torch.cat(regions, dim=0)
        # regions: n_regions, 3*n_hidden

        region_outputs = self.region_clf(regions)
        # shape of region_labels: (n_regions, n_classes)

        return region_outputs, sentence_outputs


def attention(query,key,value,mask=None,norm=True,dropout=None):
    # compute scaled dot prodcut attention
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))
    if norm:
        scores = scores/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==1,-1e9)
    p_attn = F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn = F.dropout(p_attn,p=dropout)
    return torch.matmul(p_attn,value),p_attn


class RegionCLF(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.input_dim = input_dim
        self.region_repr = CatRepr()
#        self.repr_dim = 3 * input_dim
        self.repr_dim = input_dim
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.repr_dim, n_classes),
        )

    def forward(self, data_list):
        data_repr = self.region_repr(data_list)
        # data_repr (batch_size, repr_dim)
        return self.fc(data_repr)
        # (batch_size, n_classes)


class CatRepr(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data_list):
        # shape of data_list: list(batch_size, *input_len, input_dim)
#        cat_regions = [torch.cat([hidden[0], torch.mean(hidden, dim=0), hidden[-1]], dim=-1).view(1, -1)
#                       for hidden in data_list]
        cat_regions = [torch.mean(hidden, dim=0).view(1, -1)
                       for hidden in data_list]
#        cat_regions = [torch.cat([hidden[0], hidden[-1]], dim=-1).view(1, -1)
#                       for hidden in data_list]
        cat_out = torch.cat(cat_regions, dim=0)
        # regions (batch_size, 3*input_dim)
        return cat_out


class CharLSTM(nn.Module):

    def __init__(self, n_chars, embedding_size, hidden_size, lstm_layers=1, bidirectional=True):
        super().__init__()
        self.n_chars = n_chars
        self.embedding_size = embedding_size
        self.n_hidden = hidden_size * (1 + bidirectional)

        self.embedding = nn.Embedding(n_chars, embedding_size, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True,
        )

    def sent_forward(self, words, lengths, indices):
        sent_len = words.shape[0]
        # words shape: (sent_len, max_word_len)

        embedded = self.embedding(words)
        # in_data shape: (sent_len, max_word_len, embedding_dim)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        _, (hn, _) = self.lstm(packed)
        # shape of hn:  (n_layers * n_directions, sent_len, hidden_size)

        hn = hn.permute(1, 0, 2).contiguous().view(sent_len, -1)
        # shape of hn:  (sent_len, n_layers * n_directions * hidden_size) = (sent_len, 2*hidden_size)

        # shape of indices: (sent_len, max_word_len)
        hn[indices] = hn  # unsort hn
        # unsorted = hn.new_empty(hn.size())
        # unsorted.scatter_(dim=0, index=indices.unsqueeze(-1).expand_as(hn), src=hn)
        return hn

    def forward(self, sentence_words, sentence_word_lengths, sentence_word_indices):
        # sentence_words [batch_size, *sent_len, max_word_len]
        # sentence_word_lengths [batch_size, *sent_len]
        # sentence_word_indices [batch_size, *sent_len, max_word_len]

        batch_size = len(sentence_words)
        batch_char_feat = torch.nn.utils.rnn.pad_sequence(
            [self.sent_forward(sentence_words[i], sentence_word_lengths[i], sentence_word_indices[i])
             for i in range(batch_size)], batch_first=True)

        return batch_char_feat
        # (batch_size, sent_len, 2 * hidden_size)


def main():
    pass


if __name__ == '__main__':
    main()
