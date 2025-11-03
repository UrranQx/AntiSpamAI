# models/cnn_lstm.py
import torch
import torch.nn as nn


class CNNLSTMSpamClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=128,
                 filter_sizes=[3, 4, 5], lstm_hidden=128, dropout=0.5):
        super(CNNLSTMSpamClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        self.lstm = nn.LSTM(
            num_filters * len(filter_sizes),
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)

        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)

        cat = cat.unsqueeze(1)
        lstm_out, (hidden, cell) = self.lstm(cat)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        hidden = self.dropout(hidden)
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out)
