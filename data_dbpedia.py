import numpy as np
import pandas as pd

import re
from scipy.sparse import coo_matrix
import math

from sklearn.cluster import KMeans


def preprocess(raw_data, reference_data):
    # preprocessing to get the documents and set of relation and tail       
    relation_list = []

    all_docs = []
    vocab = set()
    
    new_data = []

    # get the string info for each subject
    for s in reference_data.subject:
        if (raw_data[:,0] == s).any():
            index = np.where(raw_data[:,0]==s)[0]
            new_doc = []
            new_data.append(raw_data[index])
        
            for i in index:
                new_doc.append(raw_data[i,1])
                new_doc.append(raw_data[i,2])
        
        #         print(raw_data[i,1])
        #         print(raw_data[i,0])
            all_docs.append(new_doc)
            vocab.update(new_doc)

    vocab = sorted(list(vocab))

    print(vocab[0:10])
    vocab_index = {}
    for i, w in enumerate(vocab):
        vocab_index[w] = i
        
    # get the corpus of relation and tail    
    new_corpus = []
    for doc in all_docs:
        new_doc = []
        for word in doc:
            word_idx = vocab_index[word]
            new_doc.append(word_idx)
        new_corpus.append(new_doc)

   
    # get the all new triples facts 
    new_data_df = pd.DataFrame()
    for i in range(len(new_data)):
        temp = pd.DataFrame(new_data[i])
        new_data_df = new_data_df.append(temp,ignore_index=True)
    print(new_data_df)

    return new_corpus, vocab, new_data_df

# load the raw data from directory
raw_data = pd.read_csv('data/dbpedia/triples.txt', delimiter = '\t')
raw_data = raw_data.to_numpy()
print('---------- load the data ----------')

# load the labels for evaluation
reference_data = pd.read_csv('data/dbpedia/classes.txt', delimiter = '\t')
subjects = set(raw_data[:,0])
# check if 
for i in range(len(reference_data)):
    tmpt = reference_data.subject[i]
    if tmpt not in subjects:
        reference_data = reference_data.drop(i)

# preprocessing to get the all corpus and set of relation and tail
new_corpus, vocab, new_data_df = preprocess(raw_data, reference_data)

print(len(set(reference_data.level1)))
print(set(reference_data.level1))
print(len(set(reference_data.level2)))
print(reference_data)


def word_count_func(text):
    '''
    Counts words within a string
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Number of words within a string, integer
    ''' 
    counts = dict()
    # words = text.split() #when str
    words = text # when list

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


all = {'doc_idx':[], 'word_idx':[], 'count':[]}

for doc in range(len(new_corpus)):


  words_count_dict = word_count_func(new_corpus[doc])

  for word in words_count_dict.keys():
    all['doc_idx'].append(doc)

    # word_idx = vocab.index(word)
    all['word_idx'].append(word)

    all['count'].append(words_count_dict[word])

corpus_df = pd.DataFrame(all)
corpus_df.to_csv('data/dbpedia/corpus_poseperate.txt', sep = ' ', index=False, header = False)
print(len(all['doc_idx']))
print(corpus_df)

vocab_df = pd.DataFrame(vocab)
vocab_df.to_csv('data/dbpedia/vocab_poseperate.txt', index=False, header = False)
print(vocab_df)
print(vocab_df.iloc[210])