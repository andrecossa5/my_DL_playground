"""
Pre-trained embeddings and Recurrent Neural Network (RNN) for NLP.
"""

import torch
import matplotlib
import random
import torchtext
import collections
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, IterableDataset
matplotlib.use('MacOSX')


##


# ========================== Utils

device = "cuda" if torch.cuda.is_available() else "cpu"


##


def load_dataset(ngrams=1, min_freq=1, 
                 tokenizer=torchtext.data.utils.get_tokenizer('basic_english')):
    
    print("Loading dataset...")
    train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')
    train_dataset = list(train_dataset)
    test_dataset = list(test_dataset)
    classes = ['World', 'Sports', 'Business', 'Sci/Tech']

    print('Building vocab...')
    counter = collections.Counter()
    for (_, line) in train_dataset:
        counter.update(torchtext.data.utils.ngrams_iterator(tokenizer(line),ngrams=ngrams))
    vocab = torchtext.vocab.vocab(counter, min_freq=min_freq)

    return train_dataset,test_dataset,classes,vocab


##


def to_cbow(sent_idx, window_size=2):
    pairs = []
    for i, target in enumerate(sent_idx):
        context = [
            sent_idx[j] for j in range(max(0, i-window_size), 
                                       min(len(sent_idx), i+window_size+1))
            if j != i
        ]
        if len(context) == 2 * window_size: # Only full context windows
            pairs.append((target, context))
    return pairs


##


def collate_cbow(batch):
    target = torch.tensor([lbl for lbl, _ in batch], dtype=torch.long).to(device)
    ctx = torch.tensor([feat for _, feat in batch], dtype=torch.long).to(device)
    return target, ctx


##


class CBOWDataset(IterableDataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __iter__(self):
        random.shuffle(self.pairs)
        for label, feature in self.pairs:
            yield (label, feature)


##


class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
    def forward(self, contexts):
        embs = self.embedding(contexts)     # (batch, 2*window, embed_dim)
        avg_embs = embs.mean(dim=1)         # (batch, embed_dim): average over tokens
        logits = self.linear(avg_embs)      # (batch, vocab_size)
        return logits


##


def train_epoch_embs(model,dataloader,optimizer,loss_fn):
    """
    Train the embedding model. One epoch.
    """
    model.train()
    loss_,acc,count = 0,0,0
    for y,X in dataloader:
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out,y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_+=loss
            _,pred_y = torch.max(out,1)
            acc+=(pred_y==y).sum()
            count+=X.size(0)
            if count%1000==0:
                print(f"n examples:{count}; Mean accuracy={acc.item()/count}")

    return loss_.item()/count, acc.item()/count


##


class TextEncoder:

    def __init__(self, vocab, tokenizer, unk_index=0):
        self.stoi = vocab.get_stoi()
        self.tokenizer = tokenizer
        self.unk = unk_index

    def encode(self, text: str) -> torch.LongTensor:
        """
        Turn a raw string into a 1D LongTensor of token indices.
        """
        tokens = self.tokenizer(text)
        indices = [ self.stoi.get(tok, self.unk) for tok in tokens ]
        return torch.tensor(indices, dtype=torch.long)


##


class PadCollate:

    def __init__(self, encoder: TextEncoder, pad_index: int = 0):
        self.encoder    = encoder
        self.pad_index  = pad_index

    def __call__(self, batch):
        encoded = [ self.encoder.encode(text) for _, text in batch ]
        max_len = max(t.size(0) for t in encoded)
        padded = []
        for t in encoded:
            if t.size(0) < max_len:
                pad_amount = max_len - t.size(0)
                t = F.pad(t, (0, pad_amount), value=self.pad_index)
            padded.append(t)
        features = torch.stack(padded)
        labels = torch.tensor([lbl - 1 for lbl, _ in batch], dtype=torch.long)

        return labels, features


##


class RNNClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, 
                 pretrained_embed=None, freeze_embed=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embed is not None:  
            self.embedding.weight.data.copy_(pretrained_embed.weight.data)
            if freeze_embed:
                self.embedding.weight.requires_grad = False
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        emb = self.embedding(x)             # (batch, seq_len, embed_dim)
        out, h = self.rnn(emb)              # out:(batch, seq_len, hidden_dim)
        avg = out.mean(dim=1)               # (batch, hidden_dim): average over sequence
        logits = self.fc(avg)               # (batch, num_class)
        return logits


##


def train(model, trainset, optimizer, loss_fn, collate_fn,
          batch_size, n_epochs, device, random_state, train_size):
    """
    Basic trainer loop for a classification NN.
    """
    
    rng = torch.manual_seed(random_state)
    train_size = round(len(trainset) * .8)
    valid_size = len(trainset) - train_size
    trainset, validset = random_split(trainset, [train_size, valid_size], generator=rng)
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    train_stats = {'accuracy' : [], 'loss' : []}
    val_stats = {'accuracy' : [], 'loss' : []}

    for epoch in range(n_epochs):

        model.train()
        acc, loss_, count = 0, 0, 0
        for y,X in trainloader:
            y,X = y.to(device), X.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            loss_ += loss.item() * X.size(0)
            y_pred = torch.max(out,1)[1]
            acc += (y_pred == y).sum().item()
            count += X.size(0)
        
        train_stats['accuracy'].append(acc/count) 
        train_stats['loss'].append(loss_/count)       
        
        model.eval()
        acc, loss_, count = 0, 0, 0
        with torch.no_grad():
            for y,X in validloader:
                y,X = y.to(device), X.to(device)
                out = model(X)
                loss_ += loss_fn(out, y).item() * X.size(0)
                y_pred = torch.max(out,1)[1]
                acc += (y_pred == y).sum().item()
                count += X.size(0)
        
        val_stats['accuracy'].append(acc/count) 
        val_stats['loss'].append(loss_/count)       

        tl = train_stats['loss'][epoch]
        vl = val_stats['loss'][epoch]
        ta = train_stats['accuracy'][epoch]
        va = val_stats['accuracy'][epoch]

        print(f'Epoch: {epoch}. Mean train loss: {tl: 3f}, Mean validate loss: {vl: 3f}')
        print(f'Epoch: {epoch}. Mean train accuracy: {ta: 2f}, Mean validate accuracy: {va: 2f}')

    return pd.DataFrame(train_stats), pd.DataFrame(val_stats)


##


# ========================== #

# 1. Download and pre-process (tokenize and vocabulary contruction) AG news data
train_dataset, test_dataset, classes, vocab = load_dataset(ngrams=1, min_freq=1)
vocab_size = len(vocab)
num_classes = len(classes)
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# 2. Pre-train embeddings using CBoW framework
N = round(len(train_dataset) / 3)  # number of articles to use
cbow_pairs = []
for idx, (_, text) in enumerate(train_dataset):
    tokens = list(tokenizer(text))
    sent_idx = [vocab[t] for t in tokens if t in vocab]
    cbow_pairs += to_cbow(sent_idx, window_size=2)
    if idx >= N-1:
        break

# Create IterableDataset
cbow_ds = CBOWDataset(cbow_pairs)
cbow_loader = DataLoader(cbow_ds, batch_size=256, collate_fn=collate_cbow)

# Instantiate CBoW model
embed_dim = 128
cbow_model = CBOW(vocab_size, embed_dim).to(device)

# Train CBOW model
optimizer = torch.optim.Adam(cbow_model.parameters(), lr=.01)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 2
for epoch in range(n_epochs):
    loss, acc = train_epoch_embs(cbow_model,cbow_loader,optimizer,loss_fn)
    print(f"Epoch {epoch+1}: loss={loss:.4f}, accuracy={acc:.4f}")

# Extract pretrained embedding layer
pretrained_embs = cbow_model.embedding


##


# 3. RNNClassifier (with optional pre-trained embeddings)
loss_fn = nn.CrossEntropyLoss()
encoder = TextEncoder(vocab=vocab, tokenizer=tokenizer, unk_index=0)
collate_fn = PadCollate(encoder=encoder, pad_index=0)

# 3.1. No pre-trained embeddings
model1 = (
    RNNClassifier(
        vocab_size, embed_dim, 128, num_classes
    )
    .to(device)
)
optimizer = torch.optim.Adam(model1.parameters(), lr=.001)
train(model1, train_dataset, optimizer, loss_fn, collate_fn, 16, 5, 'cpu', 0, .8)

# Train with pre-trained embeddings
model2 = (
    RNNClassifier(
        vocab_size, embed_dim, 128, num_classes, pretrained_embs
    )
    .to(device)
)
optimizer = torch.optim.Adam(model2.parameters(), lr=.001)
train(model2, train_dataset, optimizer, loss_fn, collate_fn, 16, 5, 'cpu', 0, .8)


##
