# models/cnn_1d.py
import torch
import torch.nn as nn


class CNN1DSpamClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100,
                 filter_sizes=[3, 4, 5], dropout=0.5):
        super(CNN1DSpamClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 2)

    def forward(self, text):
        embedded = self.embedding(text)  # [batch, seq_len, emb_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch, emb_dim, seq_len]

        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)
