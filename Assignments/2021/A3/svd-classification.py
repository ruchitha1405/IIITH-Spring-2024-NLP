import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import contractions
import re
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

pad_embedding = torch.zeros(1, 300)

def preprocess_text(sentences):
    tokens = []
    for  sentence in sentences:
        sentence = contractions.fix(sentence)
        sentence = sentence.lower()
        sentence = re.sub(r"http\S+", "URL", sentence)
        sentence = re.sub(r"www\S+", "URL", sentence)
        tokenizer = RegexpTokenizer(r"\w+")
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        words = tokenizer.tokenize(sentence)
        words = ["<s>"] + words + ["</s>"]
        tokens.append(words)
    words = [word for words in tokens for word in words]
    word_freq = Counter(words)
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if word_freq[tokens[i][j]] < 3:
                tokens[i][j] = "<UNK>"
    return  tokens


def pad_sentences(sentences, max_len):
    padded_sentences = []
    for sentence in sentences:
        if len(sentence) < max_len:
            padded_sentence = sentence + ['<PAD>'] * (max_len - len(sentence))
        else:
            padded_sentence = sentence[:max_len]
        padded_sentences.append(padded_sentence)
    return padded_sentences

def convert_data_to_tensor(train_data, word_embeddings, word2idx):
    sentences = train_data["Description"].tolist()
    processed_sentences = preprocess_text(sentences)

    sentence_lengths = [len(sentence) for sentence in processed_sentences]
    max_len = int(np.percentile(sentence_lengths, 90))

    padded_sentences = pad_sentences(processed_sentences, max_len)

    X = []
    for sentence in padded_sentences:
        temp = [word_embeddings[word2idx[word]] if word in word2idx else word_embeddings[word2idx['<UNK>']] for word in sentence]
        X.append(temp)

    labels = train_data["Class Index"]
    n_classes = len(labels.unique())

    y = torch.zeros(len(labels), n_classes)
    for i, label in enumerate(labels):
        y[i][label - 1] = 1
    X = np.array(X)
    y = np.array(y)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X,y,n_classes

def generate_data(train_data,test_data, word_embeddings, word2idx):
 

    X_test,y_test,n_classes = convert_data_to_tensor(test_data, word_embeddings, word2idx)
    X,y,n_classes = convert_data_to_tensor(train_data, word_embeddings, word2idx)

    train_size = int(0.8 * len(X))
    X_val, y_val = X[train_size:], y[train_size:]
    X_train, y_train = X[:train_size], y[:train_size]

    return X_train, y_train, X_val, y_val,X_test,y_test, n_classes

def create_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test):
        # Set random seed
    torch.manual_seed(42)
    train_data = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    val_data = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    test_data = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, activation='relu'):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def init_hidden(self, batch_size):
        if self.bidirectional:
            h0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(self.device)
            c0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(self.device)
        else:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return (h0, c0)

    def forward(self, x):
        batch_size = x.size(0)
        hidden_layers = self.init_hidden(batch_size)
        out, _ = self.lstm(x, hidden_layers)
        out = self.activation(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
def train_lstm(train_loader, val_loader, model, n_epochs, lr):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))

        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')

    return train_losses, val_losses



def predictions(data_loader, model):
    all_predictions = []
    all_y_true = []

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1).cpu().tolist()
            y_true = targets.argmax(dim=1).cpu().tolist()

            all_predictions.extend(predictions)
            all_y_true.extend(y_true)

    return all_predictions, all_y_true

def scores(predictions, y_true):
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, predictions)
    metrics['f1'] = f1_score(y_true, predictions, average='weighted')
    metrics['precision'] = precision_score(y_true, predictions, average='weighted')
    metrics['recall'] = recall_score(y_true, predictions, average='weighted')
    metrics['confusion_matrix'] = confusion_matrix(y_true, predictions)

    return metrics


train_data = pd.read_csv('train.csv')
train_data = train_data[:20000]
test_data = pd.read_csv('test.csv')

word_embeddings = torch.load('svd-word-vectors.pt')
word2idx = torch.load('word2idx_svd.pt')


word_embeddings = torch.tensor(word_embeddings)
word_embeddings = torch.cat((word_embeddings, torch.zeros(1, 300)), dim=0)

word2idx['<PAD>'] = len(word2idx)

X_train, y_train, X_val, y_val,X_test,y_test, n_classes = generate_data(train_data,test_data, word_embeddings, word2idx)
train_loader, val_loader, test_loader = create_loaders(X_train, y_train, X_val, y_val, X_test, y_test)


train_loader, val_loader, test_loader = create_loaders(X_train, y_train, X_val, y_val, X_test, y_test)
input_dim = 300
hidden_dim = 128
output_dim = n_classes
n_layers = 2
bidirectional = True

model = LSTM(input_dim, hidden_dim, output_dim, n_layers, bidirectional)

n_epochs = 10
lr = 0.001

train_losses, val_losses = train_lstm(train_loader, val_loader, model, n_epochs, lr)

# Get predictions for train, val, and test sets
train_predictions, train_true = predictions(train_loader, model)
val_predictions, val_true = predictions(val_loader, model)
test_predictions, test_true = predictions(test_loader, model)

# Calculate metrics for train, val, and test sets
train_metrics = scores(train_predictions, train_true)
val_metrics = scores(val_predictions, val_true)
test_metrics = scores(test_predictions, test_true)

print(train_metrics)
print(val_metrics)
print(test_metrics)

torch.save(model.state_dict(), 'svd-classification-model.pt')
