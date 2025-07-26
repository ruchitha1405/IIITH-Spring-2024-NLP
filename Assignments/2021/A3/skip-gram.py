import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import numpy as np
import random
import numpy as np
import csv
import nltk
from nltk.tokenize import RegexpTokenizer
import contractions
import re
from collections import Counter
import string


def preprocess_text(sentences):
    tokens = []
    for _, sentence in sentences:
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
                tokens[i][j] = "UNK"
    return tokens

def create_dictionaries(tokens):
    words = [word for words in tokens for word in words]
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    idx2word = {i: word for i, word in enumerate(sorted_vocab)}
    word2idx = {word: i for i, word in idx2word.items()}
    vocab =[word for word in word2idx.keys()]
    return vocab,word2idx, idx2word

def read_corpus(file_path):
    data = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = row["Class Index"]
            description = row["Description"]
            data.append((index, description))
    return data

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNS, self).__init__()
        self.vocab_size = vocab_size
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.log_sigmoid = nn.LogSigmoid()


        ## Initialize weights
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
    
    def forward(self, target_word, context_word):
        input_embeds = self.in_embed(target_word)
        output_embeds = self.out_embed(context_word)
        return input_embeds, output_embeds
    
    def criterion(self, input_embeds, output_embeds, size, num_negatives):
        # Calculate positive loss (dot product between input and output embeddings, normalized and averaged)
        positive_loss = self.log_sigmoid(torch.sum(input_embeds * output_embeds, dim=1)).squeeze().mean()
    
        # Sample negative examples and calculate negative loss
        negative_samples = torch.LongTensor(np.random.choice(self.vocab_size, size=(size, num_negatives), replace=True, p=noise_dist))
        negative_embeds = self.out_embed(negative_samples)
        negative_loss = torch.sum(self.log_sigmoid(-torch.bmm(negative_embeds, input_embeds.unsqueeze(2)).squeeze()), dim=1).mean()
        
        # Total loss is the negative sum of positive and negative losses
        total_loss = -(positive_loss + negative_loss)
        return total_loss

    

def generate_negative_samples(vocab, idx2word):
    word_freq = Counter(vocab)
    word_freq = {word: count/len(vocab) for word, count in word_freq.items()}
    noise_dist = np.array([word_freq[word] for word in idx2word.values()])
    noise_dist = noise_dist ** (3/4)
    noise_dist = noise_dist / noise_dist.sum()
    return noise_dist

def get_negative_samples(vocab_size, num_negatives,noise_dist):
    negative_samples = np.random.choice(vocab_size, size=num_negatives, replace=False,p=noise_dist)
    return negative_samples.tolist()

def subsample_words(tokens,word2idx, threshold=1e-5):
    word_counts = Counter(tokens)
    total_count = len(tokens)
    freqs = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold / freqs[word]+threshold) for word in word_counts}
    train_words = [word for word in tokens if random.random() < (1 - p_drop[word])]
    return train_words
 
# Generate training data

def generate_training_data(vocab,corpus, window_size, word2idx):
    sampled_corpus = subsample_words(vocab,word2idx)
    training_data = []
    for sentence in corpus:
        for targetword in sentence:
            if targetword in sampled_corpus:
                if targetword in word2idx:
                    idx = word2idx[targetword]
                else:
                    idx = word2idx['UNK']
                context_idx=[]
                for i in range(-window_size, window_size+1):
                    context_word_pos = idx + i
                    if context_word_pos < 0 or context_word_pos >= len(sentence) or idx == context_word_pos:
                        continue
                    context_idx.append(word2idx[sentence[context_word_pos]])
                for context_word_idx in context_idx:
                    training_data.append((idx, context_word_idx))
    return training_data

data = read_corpus("train.csv")
corpus = preprocess_text(data[:20000])
vocab,word2idx, idx2word = create_dictionaries(corpus)
noise_dist = generate_negative_samples(vocab, idx2word)
window_size = 3
training_data=generate_training_data(vocab,corpus,window_size,word2idx)


def train_skipgram(training_data,vocab_size, embedding_dim, num_epochs, batch_size, lr):
    model = SkipGramNS(vocab_size, embedding_dim)
    model.train()
    criterion = model.criterion
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        losses = []
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            input_positive=[]
            target_positive=[]

            for a,b in batch:
                target_positive.append(b)
                input_positive.append(a)
                    
            input_positive = torch.LongTensor(input_positive)
            target_positive = torch.LongTensor(target_positive)
            model.zero_grad()
            input_embeds, output_embeds = model(input_positive, target_positive)
            loss = criterion(input_embeds, output_embeds,input_positive.size(0),5)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch: {epoch+1}, Loss: {np.mean(losses)}")
    train_embeddings = model.in_embed.weight.data.numpy()
    return train_embeddings
    
embeddings=train_skipgram(training_data, len(vocab), 300, 10, 256, 0.001)
torch.save(embeddings, "skip-gram-word-vectors.pt")
torch.save(word2idx, "word2idx_sg.pt")

# embeddings = np.load("sg_embeddings.pt", allow_pickle=True).item()
# embeddings = torch.load("sg_embeddings_2.pt")
# word2idx = torch.load("word2idx_sg_2.pt")
# print(embeddings[word2idx['obesity']])