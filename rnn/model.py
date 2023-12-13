import torch
from torch import nn
import numpy as np
import json
#import tqdm
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

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


def train(n_epochs, data, labels, embed, hidden, layers):
    labels = np.array(labels)
    reviews_enc = [[word2int[word] for word in review.split()] for review in data]

    seq_length = 256
    features = pad_features(reviews_enc, pad_id=word2int['<PAD>'], seq_length=seq_length)

    # make train set
    split_id = 1000
    train_x, remain_x = features[:split_id], features[split_id:]
    train_y, remain_y = labels[:split_id], labels[split_id:]

    # make val and test set
    split_val_id = 500
    val_x, test_x = remain_x[:split_val_id], remain_x[split_val_id:]
    val_y, test_y = remain_y[:split_val_id], remain_y[split_val_id:]

    batch_size = 32

    # create tensor datasets
    trainset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    validset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    testset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # create dataloaders
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    valloader = DataLoader(validset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    vocab_size = len(word2int)
    output_size = 1
    embedding_size = embed
    hidden_size = hidden
    n_layers = layers
    dropout = 0.25

    # model initialization
    model = SentimentModel(vocab_size, output_size, hidden_size, embedding_size, n_layers, dropout)

    lr = 0.001
    criterion = nn.BCELoss()  # we use BCELoss cz we have binary classification problem
    optim = Adam(model.parameters(), lr=lr)
    grad_clip = 5
    epochs = n_epochs
    print_every = 1
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': epochs
    }
    es_limit = 15

    #epochloop = tqdm(range(epochs), position=0, desc='Training', leave=True)

    # early stop trigger
    es_trigger = 0
    val_loss_min = torch.inf
    device="cpu"

    for e in range(epochs):

        #################
        # training mode #
        #################

        model.train()

        train_loss = 0
        train_acc = 0

        for id, (feature, target) in enumerate(trainloader):
            # add epoch meta info
            #epochloop.set_postfix_str(f'Training batch {id}/{len(trainloader)}')

            # move to device
            feature, target = feature.to(device), target.to(device)

            # reset optimizer
            optim.zero_grad()

            # forward pass
            out = model(feature)

            # acc
            predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=device)
            equals = predicted == target
            acc = torch.mean(equals.type(torch.FloatTensor))
            train_acc += acc.item()

            # loss
            loss = criterion(out.squeeze(), target.float())
            train_loss += loss.item()
            loss.backward()

            # clip grad
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # update optimizer
            optim.step()

            # free some memory
            del feature, target, predicted

        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(train_acc / len(trainloader))

        ####################
        # validation model #
        ####################

        model.eval()

        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for id, (feature, target) in enumerate(valloader):
                # add epoch meta info
                #epochloop.set_postfix_str(f'Validation batch {id}/{len(valloader)}')

                # move to device
                feature, target = feature.to(device), target.to(device)
                ft = feature

                # forward pass
                out = model(feature)

                # acc
                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=device)
                equals = predicted == target
                acc = torch.mean(equals.type(torch.FloatTensor))
                val_acc += acc.item()

                # loss
                loss = criterion(out.squeeze(), target.float())
                val_loss += loss.item()

                # free some memory
                del feature, target, predicted

            history['val_loss'].append(val_loss / len(valloader))
            history['val_acc'].append(val_acc / len(valloader))

        # reset model mode
        model.train()

        # add epoch meta info
        #epochloop.set_postfix_str(
         #   f'Val Loss: {val_loss / len(valloader):.3f} | Val Acc: {val_acc / len(valloader):.3f}')

        # print epoch
        #if (e + 1) % print_every == 0:
            #epochloop.write(
             #   f'Epoch {e + 1}/{epochs} | Train Loss: {train_loss / len(trainloader):.3f} Train Acc: {train_acc / len(trainloader):.3f} | Val Loss: {val_loss / len(valloader):.3f} Val Acc: {val_acc / len(valloader):.3f}')
           # epochloop.update()

        # save model if validation loss decrease
        if val_loss / len(valloader) <= val_loss_min:
            torch.save(model.state_dict(), '/djangoProjectFinalVersion/rnn/utils/sentiment_lstm.pt')
            val_loss_min = val_loss / len(valloader)
            es_trigger = 0
       # else:
            #epochloop.write(
             #   f'[WARNING] Validation loss did not improved ({val_loss_min:.3f} --> {val_loss / len(valloader):.3f})')
            #es_trigger += 1

        # force early stop
        if es_trigger >= es_limit:
            #epochloop.write(f'Early stopped at Epoch-{e + 1}')
            # update epochs history
            history['epochs'] = e + 1
            break

    return history


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