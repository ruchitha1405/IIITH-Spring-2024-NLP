import torch
import numpy as np
import argparse

class SimilarityFinder:
    def __init__(self, embeddings_path, word2idx_path):
        self.word_vectors = torch.load(embeddings_path)
        self.word2idx = torch.load(word2idx_path)
        # print(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def most_similar(self, word, topn=10):
        if word not in self.word2idx:
            print(f"Word '{word}' not found in vocabulary.")
            return []

        word_idx = self.word2idx[word]
        if word_idx >= len(self.word_vectors):
            print(f"Word '{word}' index out of range.")
            return []

        word_vector = self.word_vectors[word_idx]
        similarities = np.dot(self.word_vectors, word_vector)
        most_similar_indices = np.argsort(similarities)[::-1][:topn]
        most_similar_words = [(self.idx2word[idx.item()],similarities[idx.item()]) for idx in most_similar_indices]
        return most_similar_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word Similarity Finder")
    parser.add_argument("-svd", action="store_true", help="Use SVD model")
    parser.add_argument("-w2v", action="store_true", help="Use Word2Vec model")
    args = parser.parse_args()

    if args.svd:
        embeddings_path = 'models/svd-word-vectors.pt'
        word2idx_path = 'models/word2idx_svd.pt'
    elif args.w2v:
        embeddings_path = 'models/skip-gram-word-vectors.pt'
        word2idx_path = 'models/word2idx_sg.pt'
    else:
        print("Please specify either -svd or -w2v option.")
        exit(1)

    similarity_finder = SimilarityFinder(embeddings_path, word2idx_path)

    word = 'company'
    similar_words = similarity_finder.most_similar(word)
    print(f"Most similar words to '{word}':")
    for word in similar_words:
        print(word)