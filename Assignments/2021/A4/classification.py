import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import contractions
from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse

print("imports done...")


torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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


def read_file(path):
    df = pd.read_csv(path)
    corpus = []
    labels = []
    for i in range(len(df)):
        x = tokenize(df.at[i, 'Description'])
        corpus.append(x)
        labels.append(int(df.at[i, 'Class Index'])-1)
    return corpus, labels


class createClassificationDataset(Dataset):
    def __init__(self, sentences,labels,word2idx,idx2word):
        self.sentences = sentences
        self.labels = labels
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.max_len = 0
        lengths = []    
        for i in range(len(sentences)):
            lengths.append(len(sentences[i]))
        self.max_len = int(np.percentile(lengths, 95))
        self.X,self.y = self.create_data()

    def create_data(self):
        X = []
        y = []
        for i in range(len(self.sentences)):
            sentence = self.sentences[i]
            if len(sentence) < self.max_len:
                sentence = sentence + ['<pad>']*(self.max_len-len(sentence))
            elif len(sentence) > self.max_len:
                sentence = sentence[:self.max_len]
            temp_x=[]
            temp_y=[]
            for word in sentence:
                if word in self.word2idx:
                    temp_x.append(self.word2idx[word])
                else:
                    temp_x.append(self.word2idx['<unk>'])
            X.append(temp_x)
            y.append(self.labels[i])
        return X,y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]),torch.tensor(self.y[idx])


class DownStreamClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, bidirectional, activation_fn,type):
        super(DownStreamClassifier, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        if type == "trainable":
            ls = torch.rand(3)
            ls/= ls.sum()
            self.lambdas = nn.Parameter(ls, requires_grad=True)
        elif type== "fixed":
            ls = torch.rand(3)
            ls/= ls.sum()
            self.lambdas = nn.Parameter(ls, requires_grad=False)
        elif type == "learnable":
            self.learnable_function(embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers    
        self.bidirectional = bidirectional
        self.activation_fn = activation_fn
        self.type = type

    def learnable_function(self,embedding_dim):
        self.functional_layer = nn.Linear(embedding_dim*3,embedding_dim)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=0.36)
        nn.init.xavier_uniform_(self.functional_layer.weight)

    def init_hidden(self, x):
        h0 = torch.zeros(self.num_layers * 2 if self.bidirectional else 1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2 if self.bidirectional else 1, x.size(0), self.hidden_dim).to(x.device)
        return h0, c0

    def forward(self, e0, h1, h2):
        if self.type=="learnable":
            X = torch.cat([e0, h1, h2], dim=2)
            X = self.functional_layer(X)
            X = self.activation_fn(X)
            X = self.dropout(X)

        else :
            X = torch.stack([w * t for w, t in zip(self.lambdas, [e0, h1, h2])]).sum(dim=0)
        h0, c0 = self.init_hidden(X)
        output, _ = self.lstm(X, (h0, c0))
        output = self.activation_fn(output)
        output = output[:, -1, :]
        output = self.fc(output)
        return output

def train_model(classifier, train_loader, val_loader, optimizer, criterion, num_epochs,model):
    """Train the downstream classifier."""
    start = time.time()
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    train_losses =[]
    val_losses = []
    
    for epoch in range(num_epochs):
        classifier.train()
        correct_predictions = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        training_loss = 0
        for batch,(inputs, labels) in enumerate(train_loader):
            inputs , labels = inputs.to(device), labels.to(device)

            embeds_forward = model.embedding(inputs)
            layer1_forward,_ =model.lstm_forward_1(embeds_forward)
            layer2_forward,_ = model.lstm_forward_2(layer1_forward)

            inputs_reverse = torch.flip(inputs, [1])

            embeds_backward=model.embedding(inputs_reverse)
            layer1_backward,_ = model.lstm_backward_1(embeds_backward)
            layer2_backward,_ = model.lstm_backward_2(layer1_backward)

            embeds = torch.cat((embeds_forward, embeds_backward), dim=2)
            layer1 = torch.cat((layer1_forward, layer1_backward), dim=2)
            layer2 = torch.cat((layer2_forward, layer2_backward), dim=2)
            input_features = torch.cat((embeds, layer1, layer2), dim=2)
            
            # Forward pass through the classifier
            outputs = classifier(embeds, layer1, layer2)
            
            # Calculate predictions
            _, preds = torch.max(outputs, 1)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            training_loss += loss.item()
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Collect labels and predictions for F1 score calculation
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            if batch%1000==1:
                print(f'Epoch: {epoch+1}, Step: {batch}, Loss: {loss.item()}')
        
        # Calculate train accuracy and F1 score
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_accuracies.append(train_accuracy)
        train_f1_score = f1_score(all_labels, all_preds, average='weighted')
        train_f1_scores.append(train_f1_score)
        train_losses.append(training_loss/len(train_loader))

        # Validation
        classifier.eval()
        correct = 0
        total = 0
        val_all_labels = []
        val_all_preds = []
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs , labels = inputs.to(device), labels.to(device)

                embeds_forward = model.embedding(inputs)
                layer1_forward,_ =model.lstm_forward_1(embeds_forward)
                layer2_forward,_ = model.lstm_forward_2(layer1_forward)

                inputs_reverse = torch.flip(inputs, [1])

                embeds_backward=model.embedding(inputs_reverse)
                layer1_backward,_ = model.lstm_backward_1(embeds_backward)
                layer2_backward,_ = model.lstm_backward_2(layer1_backward)

                embeds = torch.cat((embeds_forward, embeds_backward), dim=2)
                layer1 = torch.cat((layer1_forward, layer1_backward), dim=2)
                layer2 = torch.cat((layer2_forward, layer2_backward), dim=2)
                input_features = torch.cat((embeds, layer1, layer2), dim=2)

                # Forward pass through the classifier
                outputs = classifier(embeds, layer1, layer2)

                # Calculate predictions
                _, preds = torch.max(outputs, 1)

                # Calculate loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
               
                # Collect labels and predictions for F1 score calculation
                val_all_labels.extend(labels.cpu().tolist())
                val_all_preds.extend(preds.cpu().tolist())


            val_accuracy = accuracy_score(val_all_labels, val_all_preds)
            val_accuracies.append(val_accuracy)
            val_f1_score = f1_score(val_all_labels, val_all_preds, average='weighted')
            val_f1_scores.append(val_f1_score)
            val_losses.append(val_loss/len(val_loader))   
            t = time.time() - start      
            minutes, seconds = divmod(t, 60)
 
            print(f' ({minutes}m {int(seconds)}s) Epoch: {epoch+1}/{num_epochs}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy} , Train Loss: {train_losses[-1]} , Val Loss: {val_losses[-1]}' )
    
    return train_accuracies, val_accuracies, train_f1_scores, val_f1_scores,train_losses,val_losses,all_labels,all_preds,val_all_labels,val_all_preds


def test_model(classifier, test_loader, model):
    classifier.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch, (inputs, labels) in enumerate(test_loader):
            inputs , labels = inputs.to(device), labels.to(device)

            embeds_forward = model.embedding(inputs)
            layer1_forward,_ =model.lstm_forward_1(embeds_forward)
            layer2_forward,_ = model.lstm_forward_2(layer1_forward)

            inputs_reverse = torch.flip(inputs, [1])

            embeds_backward=model.embedding(inputs_reverse)
            layer1_backward,_ = model.lstm_backward_1(embeds_backward)
            layer2_backward,_ = model.lstm_backward_2(layer1_backward)

            embeds = torch.cat((embeds_forward, embeds_backward), dim=2)
            layer1 = torch.cat((layer1_forward, layer1_backward), dim=2)
            layer2 = torch.cat((layer2_forward, layer2_backward), dim=2)
            input_features = torch.cat((embeds, layer1, layer2), dim=2)

            # Forward pass through the classifier
            outputs = classifier(embeds, layer1, layer2)

            # Calculate predictions
            _, preds = torch.max(outputs, 1)

            # Collect labels and predictions for F1 score calculation
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
        test_accuracy = accuracy_score(all_labels, all_preds)
        test_f1_score = f1_score(all_labels, all_preds, average='weighted')
    return test_accuracy, test_f1_score, all_labels, all_preds



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification model") 
    parser.add_argument("-f", action="store_true", help="using frozen lambdas")
    parser.add_argument("-t", action="store_true", help="using trainable lambdas")
    parser.add_argument("-l", action="store_true", help="using learnable function")
    args = parser.parse_args()
    train_corpus, train_labels = read_file('train.csv')
    num_classes = len(set(train_labels))
    test_corpus, test_labels = read_file('test.csv')

    print("preprocessing done")
    word2idx = torch.load('word2idx.pt')
    idx2word = torch.load('idx2word.pt')
    dataset = createClassificationDataset(train_corpus, train_labels, word2idx, idx2word)

    train_length = int(0.8 * len(dataset))
    val_length = len(dataset) - train_length
    train_data, val_data = random_split(dataset, [train_length, val_length])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    print("dataset created for train,val")

    model = torch.load('bilstm.pt').to(device)
    activation_fn = nn.ReLU()

    print("models loaded")

    if args.f:
        classifier = DownStreamClassifier(600,300,1,num_classes,True,activation_fn,"fixed").to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        print("classifier created")

    elif args.t:
        classifier = DownStreamClassifier(600,300,1,num_classes,True,activation_fn,"trainable").to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        print("classifier created")

    elif args.l:
        classifier = DownStreamClassifier(600,300,1,num_classes,True,activation_fn,"learnable").to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(list(classifier.parameters())+list(classifier.functional_layer.parameters()), lr=0.001)
        print("classifier created")
    
    else:
        print("Please specify either -f or -t or -l option.")
        exit(1)

    num_epochs = 10
    print("training....")
    train_accuracies, val_accuracies, train_f1_scores, val_f1_scores,train_loss,val_loss,train_labels,train_preds,val_labels,val_preds = train_model(classifier, train_loader, val_loader, optimizer, criterion, num_epochs,model)
    

    print("creating test dataset")
    test_dataset = createClassificationDataset(test_corpus, test_labels, word2idx, idx2word)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

 


    print("testing....")
    test_accuracy,test_f1_score,test_labels,test_preds=test_model(classifier, test_loader, model)
    print("Test Accuracy: ", test_accuracy, "Test f1 score: ",test_f1_score)

    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix:")
    print(cm)
    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    if args.t:
        print("trained lambda: ", classifier.lambdas)
        # Save the model
        torch.save(classifier, 'classifier_t.pt')
        print("model saved")
    elif args.f:
        print("trained lambda: ", classifier.lambdas)
        torch.save(classifier, 'classifier_f.pt')
        print("model saved")
    elif args.l:
        print("learnable function parameters: ", classifier.functional_layer.weight)
        torch.save(classifier, 'classifier_l.pt')
        print("model saved")
    else:
        print()