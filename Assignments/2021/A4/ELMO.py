import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import contractions
from collections import Counter
import time

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def tokenize(text):

    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+', '<url>', text)
    text = re.sub(r'www\S+', '<url>', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    tokens = text.split()  
    tokens = ['<s>'] + tokens + ['</s>']
    return tokens


df = pd.read_csv('train.csv')
corpus = []
labels = []
for i in range(len(df)):
    x = tokenize(df.at[i, 'Description'])
    corpus.append(x)
    labels.append(int(df.at[i, 'Class Index'])-1)


print("preprocessing done")

class createDataset(Dataset):
    def __init__(self, corpus=None, labels=None,vocab=None, word2idx=None, idx2word=None):
        self.corpus = corpus
        self.labels = labels
        if vocab is None:
            self.word2idx, self.idx2word = self.create_vocab()
        else:
            self.word2idx, self.idx2word = word2idx, idx2word
        self.sentences = self.pad_sentences()
        self.forward_labels, self.forward_targets = self.forward()
        self.backward_labels, self.backward_targets = self.backward()


    def create_vocab(self):
        word_counts = Counter(word for sentence in self.corpus for word in sentence)
        word2idx = {"<s>": 0, "</s>": 1, "<unk>": 2, "<pad>": 3}
        idx2word = {0: "<s>", 1: "</s>", 2: "<unk>", 3: "<pad>"}

        for word, count in word_counts.items():
            if count < 3:
                word = "<unk>"
            if word not in word2idx:
                idx = len(word2idx)
                word2idx[word] = idx
                idx2word[idx] = word

        return word2idx, idx2word
    
    def pad_sentences(self):
        lengths = [len(sentence) for sentence in self.corpus]
        max_length = np.percentile(lengths, 95)
        sentences = []
        for sentence in self.corpus:
            sentence = [self.word2idx[word] if word in self.word2idx else self.word2idx["<unk>"] for word in sentence]
            if len(sentence) < max_length:
                sentence = sentence + [self.word2idx["<pad>"]] * (int(max_length) - len(sentence))
            else:
                sentence = sentence[:int(max_length)]
            sentences.append(sentence)
        return sentences
    

    def forward(self):
        # Generate X and y for the forward model
        X = [sentence[:-1] for sentence in self.sentences]  # Each sentence without the last token
        y = [sentence[1:] for sentence in self.sentences]   # Each sentence without the first token        
        return X, y

    def backward(self):
        # Generate X and y for the backward model
        X = [sentence[::-1][:-1] for sentence in self.sentences]  # Reverse each sentence and remove the last token
        y = [sentence[::-1][1:] for sentence in self.sentences]   # Reverse each sentence and remove the first token
        return X, y

    def __len__(self):
        return len(self.forward_labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.forward_labels[idx]), torch.tensor(self.forward_targets[idx]), torch.tensor(self.backward_labels[idx]), torch.tensor(self.backward_targets[idx])

    
class elmo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(elmo, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_forward_1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_forward_2 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_forward = nn.Linear(hidden_dim, vocab_size)
        self.lstm_backward_1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_backward_2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_backward = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_forward, x_backward):
        forward_embeds = self.embedding(x_forward)
        forward_lstm_1, _ = self.lstm_forward_1(forward_embeds)
        forward_lstm_2, _ = self.lstm_forward_2(forward_lstm_1)
        forward_outputs = self.fc_forward(forward_lstm_2)

        backward_embeds = self.embedding(x_backward)
        backward_lstm_1, _ = self.lstm_backward_1(backward_embeds)
        backward_lstm_2, _ = self.lstm_backward_2(backward_lstm_1)
        backward_outputs = self.fc_backward(backward_lstm_2)

        return forward_outputs, backward_outputs


def train(model, data_loader, vocab_size, num_epochs):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    start = time.time()

    for epoch in range(num_epochs):
        total_loss=0
        model.train()
        for batch, (forward_labels, forward_targets, backward_labels, backward_targets) in enumerate(data_loader):
            forward_labels, forward_targets = forward_labels.to(device), forward_targets.to(device)
            backward_labels, backward_targets = backward_labels.to(device), backward_targets.to(device)
            optimizer.zero_grad()
            forward_outputs, backward_outputs = model(forward_labels, backward_labels)
            forward_loss = criterion(forward_outputs.view(-1, vocab_size), forward_targets.view(-1))
            backward_loss = criterion(backward_outputs.view(-1, vocab_size), backward_targets.view(-1))
            loss = forward_loss + backward_loss
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            if batch % 1000 == 1:
                print(f'Epoch: {epoch+1}, Step: {batch},Loss: {loss.item()}')
        
        losses.append(total_loss/len(data_loader))
        t = time.time() - start

        print(f' ({int(t/60)}m {int(t%60)}s) Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')

    return losses


dataset = createDataset(corpus, labels)
word2idx, idx2word = dataset.word2idx, dataset.idx2word
print("dataset created")

model = elmo(len(word2idx), 300, 300)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("data loader done")


losses = train(model, data_loader, len(word2idx), 10)
torch.save(model, 'bilstm.pt')
torch.save(word2idx, 'word2idx.pt')
torch.save(idx2word, 'idx2word.pt')

print("model saved")



