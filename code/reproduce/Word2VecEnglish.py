'''
import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

#%load_ext tensorboard

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split())
print(len(tokens))
'''
'''
# Python program to generate word vectors using Word2Vec
# code is originally from:  https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
import nltk
nltk.download('punkt')
# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
 
warnings.filterwarnings(action = 'ignore')
 
import gensim
from gensim.models import Word2Vec
 
#  Reads ‘alice.txt’ file
sample = open("Alice.txt")
s = sample.read()
 
# Replaces escape character with space
f = s.replace("\n", " ")
 
data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []
     
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
 
    data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1,
                              vector_size = 300, window = 5)
 
# Print results
print("Cosine similarity between 'alice' " +
               "and 'wonderland' - CBOW : ",
    model1.wv.similarity('alice', 'wonderland'))
     
print("Cosine similarity between 'alice' " +
                 "and 'machines' - CBOW : ",
      model1.wv.similarity('alice', 'machines'))
 
# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 300,
                                             window = 5, sg = 1)
 
# Print results
print("Cosine similarity between 'alice' " +
          "and 'wonderland' - Skip Gram : ",
    model2.wv.similarity('alice', 'wonderland'))
     
print("Cosine similarity between 'alice' " +
            "and 'machines' - Skip Gram : ",
      model2.wv.similarity('alice', 'machines'))


# now analyzing the rw.txt data
rw = open('rw.txt','r')
rw = rw.readlines()
for line in rw:
    l = line.split()
    word1 = l[0]
    word2 = l[1]
    avg = l[2]
    
    
#model =  gensim.models.Word2Vec.load("wiki_300_50_word2vec.model")
'''


'''
### Most useful for 5.1 so far!!!

import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from scipy.stats import spearmanr

print('loading...')
model = gensim.models.KeyedVectors.load("FrModel.bin")
print('gotcha!')

rw = open('RG65_FR.txt','r')
rw = rw.readlines()
words = list()
for line in rw:
    if line != '':
        l = line.split('\t')
        word1 = l[0]
        word2 = l[1]
        avg = l[2]
        words.append([word1,word2,avg])
print('ready to calculate!!!!')
cosine = list()
avg = list()
print('length:',len(words))
num = 0
for item in words:
    try:
        cos = model.similarity(item[0], item[1])
        print("Cosine similarity between '"+item[0]+" and '"+item[1]+"':", cos)
        cosine.append(cos)
        avg.append(item[2])
        num += 1
    except:
        print("'"+item[0]+"' or '"+item[1]+"' not in vocab'")

print('evaluated:',num)
r = spearmanr(cosine, avg)
print(r[0])

print('done')

'''

'''
import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
#file = "enwiki_20180420_300d.txt"
print('loading...')
model = Word2Vec.load("wiki_300_50_word2vec.model")
print('gotcha!')
#model.save("EnModel.bin")
#print('saved')
#print(model.most_similar("apple"))
#print(model.wv.most_similar("apple"))
#model.save("myModel.model")

# now analyzing the rw.txt data
rw = open('rw.txt','r')
rw = rw.readlines()
words = list()
for line in rw:
    l = line.split('\t')
    word1 = l[0]
    word2 = l[1]
    avg = l[2]
    words.append([word1,word2,avg])

for item in words:
    try:
        print("Cosine similarity between '"+item[0]+
            " and '"+item[1]+"':", model.wv.similarity(item[0], item[1]))
    except:
        print(item[0],'or',item[1],'not in vocab')
'''

### for 5.2 calculations
import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

def getAverage_for_5_2_data(bin, test_name, lang):
    accuracy = 0
    total_eval = 0
    total_lines = 0
    print('loading',bin,'...')
    model = gensim.models.KeyedVectors.load(bin)
    print('gotcha!')

    file = open(test_name+'.txt','r')
    f = file.readlines()
    file.close()
    
    words = list()
    for line in f:
        #print(line)
        total_lines += 1
        if line != '':
            l = line.split()
            word1 = l[0].lower().strip()
            word2 = l[1].lower().strip()
            word3 = l[2].lower().strip()
            word4 = l[3].lower().strip()
            # check that word exists, then write to the files
            # word 1
            if word1 in model and word2 in model and word3 in model and word4 in model:
                total_eval += 1
                '''
                print(line)
                #prediction_vector = model[word1] - model[word2] + model[word3]
                #print(prediction_vector)
                #print('now for the similar one:')
                similar = model.most_similar(positive=[word2,word3],negative=[word1],topn=1)
                print(similar)
                if str(similar[0][0]).strip() == word4.strip():
                    accuracy += 1
                    print('found a match!')
                '''
    print(accuracy,'out of',total_eval,'are correct')
    print('total lines:',total_lines)

def format_German(file, new_name):
    f = open(file,'r')
    lines = f.readlines()
    new_lines = list()
    for line in lines:
        l = line.split(':')
        new_lines.append('\t'.join(l))
    f.close()
    
    f = open(new_name,'w')
    for line in new_lines:
        f.write(line.lower())
    f.close()
    
getAverage_for_5_2_data('EnModel.bin','mikolov-syntactic','en')




