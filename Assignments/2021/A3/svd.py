import numpy as np
import csv
import nltk
from nltk.tokenize import RegexpTokenizer
import contractions
import re
from collections import Counter
import string
import torch


class CooccurenceMatrix:
    def __init__(self, corpus, context_window=1):
        self.corpus = corpus
        self.context_window = context_window
        self.vocab = {}
        self.word2idx={}
        self.idx2word={}
        self.cooccurence_matrix = self.build_cooccurence_matrix()

    def build_cooccurence_matrix(self):

        vocab_size = 0
        for words in self.corpus:
            for i in range(len(words)):
                if words[i] not in self.vocab:
                    self.vocab[words[i]] = vocab_size
                    vocab_size += 1

        for word in self.vocab:
            self.word2idx[word] = self.vocab[word]
            self.idx2word[self.vocab[word]] = word

        cooccurence_matrix = np.zeros((vocab_size, vocab_size))
        for sentence in self.corpus:
            for i, word in enumerate(sentence):
                start = max(0, i - self.context_window)
                end = min(len(sentence), i + self.context_window)
                context_words = sentence[start:i] + sentence[i + 1 : end + 1]
                for context in context_words:
                    if self.vocab[word] != self.vocab[context]:
                        cooccurence_matrix[self.vocab[word]][self.vocab[context]] += 1
                        cooccurence_matrix[self.vocab[context]][self.vocab[word]] += 1
                    else:
                        cooccurence_matrix[self.vocab[word]][self.vocab[context]] += 1

        self.cooccurence_matrix=cooccurence_matrix
        return cooccurence_matrix

    def svd(self, embedding_size=100):
        U,s,Vt = np.linalg.svd(self.cooccurence_matrix)
        word_vectors = U[:, :embedding_size]
        self.word_vectors = word_vectors
        return word_vectors

    def get_vector(self, word):
        if word not in self.vocab:
            return None
        else:
            return self.word_vectors[self.vocab[word]]

    def most_similiar(self, word, topn=5):
        if word not in self.vocab:
            return []
        word_vector = self.get_vector(word)
        if word_vector is None:
            return []
        similiarity = np.dot(self.word_vectors, word_vector)
        most_similiar_words = np.argsort(similiarity)[::-1][:topn]
        print(most_similiar_words)
        return [(self.idx2word[word], similiarity[word]) for word in most_similiar_words]

    def save_embeddings(self, path):
        torch.save(self.word_vectors,path)

def read_corpus(file_path):
    data = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = row["Class Index"]
            description = row["Description"]
            data.append((index, description))
    return data


def preprocess_text(sentences):
    tokens = []
    for _,sentence in sentences:
       sentence = contractions.fix(sentence)
       sentence = sentence.lower()
       sentence = re.sub(r'http\S+', 'URL', sentence)
       sentence = re.sub(r'www\S+', 'URL', sentence)
       tokenizer = RegexpTokenizer(r'\w+')
       sentence = sentence.translate(str.maketrans('', '', string.punctuation))
       words = tokenizer.tokenize(sentence)
       words = ['<s>'] + words + ['</s>']
       tokens.append(words)
    words = [word for words in tokens for word in words]
    word_freq = Counter(words)
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if word_freq[tokens[i][j]] < 3:
                tokens[i][j] = '<UNK>'
    return tokens




data = read_corpus("train.csv")
corpus = preprocess_text(data[0:20000])
cooccurence_matrix = CooccurenceMatrix(corpus, context_window=1)
print(len(cooccurence_matrix.vocab))
print(cooccurence_matrix.cooccurence_matrix.shape)

word_vectors = cooccurence_matrix.svd(embedding_size=300)
print(cooccurence_matrix.most_similiar('obesity'))
cooccurence_matrix.save_embeddings('svd-word-vectors.pt')
torch.save(cooccurence_matrix.vocab,'word2idx_svd.pt')
# np.save('word2idx.npy', cooccurence_matrix.word2idx)
# np.save('idx2word.npy', cooccurence_matrix.idx2word)






