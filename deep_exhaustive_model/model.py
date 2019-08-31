# coding: utf-8

import torch
import torch.nn as nn
import numpy as np


class ExhaustiveModel(nn.Module):

    def __init__(self, hidden_size, n_tags, max_region, embedding_url=None, bidirectional=True, lstm_layers=1,
                 n_embeddings=None, embedding_dim=None, freeze=False, char_feat_dim=100):
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
            n_chars=100,
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
        self.max_region = max_region
        self.n_hidden = (1 + bidirectional) * hidden_size

        self.region_clf = nn.Sequential(
            nn.ReLU(),
            nn.Linear(3*self.n_hidden, n_tags),
            # nn.Softmax(),
        )

    def forward(self, sentences, sentence_lengths, sentence_words, sentence_word_lengths,
                sentence_word_indices):

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

        # word_repr = self.dropout(word_repr)

        packed = nn.utils.rnn.pack_padded_sequence(word_repr, sentence_lengths, batch_first=True)
        out, (hn, _) = self.lstm(packed)

        max_sent_len = sentences.shape[1]
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=max_sent_len, batch_first=True)
        # unpacked (batch_size, max_sent_len, n_hidden)
        unpacked = unpacked.transpose(0, 1)
        # unpacked (max_sent_len, batch_size, n_hidden)
        # shape of hn:  (n_layers * n_directions, batch_size, hidden_size)

        max_len = sentence_lengths[0]
        regions = list()
        for region_size in range(1, self.max_region + 1):
            for start in range(0, max_len - region_size + 1):
                end = start + region_size
                regions.append(torch.cat([unpacked[start], torch.mean(unpacked[start:end], dim=0),
                                          unpacked[end - 1]], dim=-1))
#                regions.append(torch.mean(unpacked[start:end],dim=0))
#                regions.append(torch.cat([unpacked[start],unpacked[end-1]],dim=-1))
                # shape of each region: (batch_size, 3 * n_hidden)
        output = torch.stack([self.region_clf(region) for region in regions], dim=-1)
        # shape of each region_clf output: (batch_size, n_classes)
        # shape of output: (batch_size, n_classes, n_regions)
        return output


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
