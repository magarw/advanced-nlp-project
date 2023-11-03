from gensim.models.keyedvectors import KeyedVectors
import os

if __name__ == '__main__':
    if os.path.exists(f"word2vec.wordvectors"):
        wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
        print("word vectors loaded")
        vector = wv['skdvdslkj']
        print(vector)
