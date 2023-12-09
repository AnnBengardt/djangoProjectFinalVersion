import torch
from torch import nn
import numpy as np
import json

with open('/djangoProjectFinalVersion/rnn/utils/word2int.json') as f:
    word2int = json.load(f)


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, output_size, hidden_size=128, embedding_size=256, n_layers=2, dropout=0.2):
        super(SentimentModel, self).__init__()

        # embedding layer is useful to map input into vector representation
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # LSTM layer preserved by PyTorch library
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # Linear layer for output
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid layer cz we will have binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # convert feature to long
        x = x.long()

        # map input to vector
        x = self.embedding(x)

        # pass forward to lstm
        o, _ = self.lstm(x)

        # get last sequence output
        o = o[:, -1, :]

        # apply dropout and fully connected layer
        o = self.dropout(o)
        o = self.fc(o)

        # sigmoid
        o = self.sigmoid(o)

        return o

def pad_features(reviews, pad_id, seq_length=128):
    # features = np.zeros((len(reviews), seq_length), dtype=int)
    features = np.full((len(reviews), seq_length), pad_id, dtype=int)

    for i, row in enumerate(reviews):
        # if seq_length < len(row) then review will be trimmed
        features[i, :len(row)] = np.array(row)[:seq_length]

    return features


def predict(text):
    vocab_size = len(word2int)
    output_size = 1
    embedding_size = 256
    hidden_size = 512
    n_layers = 2
    dropout = 0.25

    # model initialization
    model = SentimentModel(vocab_size, output_size, hidden_size, embedding_size, n_layers, dropout)
    print("loading weights...")
    model.load_state_dict(torch.load("/djangoProjectFinalVersion/rnn/utils/lstm.pt"))
    print("weights loaded")
    model.eval()

    reviews_enc = [[word2int[word] for word in text.split()]]

    seq_length = 256
    features = pad_features(reviews_enc, pad_id=word2int['<PAD>'], seq_length=seq_length)

    assert len(features) == len(reviews_enc)
    assert len(features[0]) == seq_length

    with torch.no_grad():
        print("starting prediction...")
        out = model(torch.tensor(features))
        predicted = 1 if out > 0.5 else 0

    return predicted