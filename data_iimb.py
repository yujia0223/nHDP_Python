from rdflib import Graph

import pandas as pd
import numpy as np

import os



g = Graph()
# g.parse('data/IIMB_SMALL/000/onto.owl')
g.parse('data/IIMB_LARGE/000/onto.owl')

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


facts = []
for index, (s,p,o) in enumerate(g):
    sub = remove_prefix(s,'http://oaei.ontologymatching.org/2010/IIMBDATA/')
    sub = remove_prefix(sub, 'http://oaei.ontologymatching.org/2010/IIMBTBOX/')

    pre = remove_prefix(p, 'http://oaei.ontologymatching.org/2010/IIMBTBOX/')
    pre = remove_prefix(pre, 'http://www.w3.org/')

    obj = remove_prefix(o, 'http://oaei.ontologymatching.org/2010/IIMBDATA/')
    obj = remove_prefix(obj, 'http://oaei.ontologymatching.org/2010/IIMBTBOX/')
    obj = remove_prefix(obj, 'http://www.w3.org/2001/XMLSchema')


    # if 'type' not in p: # add type info
    if 'article' not in p:
        facts.append([sub,pre,obj])

facts_df = pd.DataFrame(facts, columns=['s','p','o'])
print(facts_df)
print(set(facts_df['s']))
print(len(set(facts_df['s'])))


def preprocess(raw_data): # delete reference for a while
    # preprocessing to get the documents and set of relation and tail       
    relation_list = []

    all_docs = []
    vocab = set()
    
    new_data = []

    # get the string info for each subject
    for s in set(raw_data.s):
        if (raw_data['s'] == s).any():
            index = np.where(raw_data['s']==s)[0]
            new_doc = []
            new_data.append(raw_data.iloc[index])
        
            for i in index:
                # new_doc.append(raw_data.iloc[i,1]) # property
                new_doc.append(raw_data.iloc[i,2]) # object
        
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
    # print(new_data_df)

    return new_corpus, vocab, new_data_df

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


# preprocessing to get the all corpus and set of relation and tail
new_corpus, vocab, new_data_df = preprocess(facts_df)
print(len(vocab[-10:]))
print(len(vocab))
print(new_data_df)



all = {'doc_idx':[], 'word_idx':[], 'count':[]}

for doc in range(len(new_corpus)):


    words_count_dict = word_count_func(new_corpus[doc])

    for word in words_count_dict.keys():
        all['doc_idx'].append(doc)

        # word_idx = vocab.index(word)
        all['word_idx'].append(word)

        all['count'].append(words_count_dict[word])


corpus_df = pd.DataFrame(all)
# corpus_df.to_csv('data/iimb/corpus_poseperate.txt', sep = ' ', index=False, header = False)
# corpus_df.to_csv('data/iimb_l/corpus_ot.txt', sep = ' ', index=False, header = False)
print(len(all['doc_idx']))
print(corpus_df)
print(corpus_df[corpus_df['count'] > 1])

vocab_df = pd.DataFrame(vocab)
# vocab_df.to_csv('data/iimb/vocab_poseperate.txt', index=False, header = False)
# vocab_df.to_csv('data/iimb_l/vocab_ot.txt', index=False, header = False)
print(vocab_df)
# print(vocab_df.iloc[210])

# to check the original data
all_subjects = list(set(facts_df.s))
print('the 1st document is: \n')
print(new_data_df[new_data_df.s == all_subjects[0]])
print('the 2nd document is: \n')
print(new_data_df[new_data_df.s == all_subjects[1]])
print('the 99th document is: \n')
print(new_data_df[new_data_df.s == all_subjects[99]])