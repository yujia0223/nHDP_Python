import numpy as np
import pandas as pd

def preprocess(raw_data, reference_data):
    # preprocessing to get the documents and set of relation and tail       
    relation_list = []
    all_doc = []
    vocab = set()
    all_tail_doc = []
    tail_vocab = set()
    new_data = []

    for s in reference_data.subject:
        if (raw_data[:,0] == s).any():
            index = np.where(raw_data[:,0]==s)[0]
            new_doc = []
            new_tail_doc = []
            new_data.append(raw_data[index])
        
            for i in index:
                new_doc.append(raw_data[i,1])
                new_tail_doc.append(raw_data[i,2])
        
        #         print(raw_data[i,1])
        #         print(raw_data[i,0])
            all_doc.append(new_doc)
            all_tail_doc.append(new_tail_doc)
            vocab.update(new_doc)
            tail_vocab.update(new_tail_doc)


    vocab = sorted(list(vocab))
    tail_vocab = sorted(list(tail_vocab))

    print(vocab[0:10])
    vocab_index = {}
    for i, w in enumerate(vocab):
        vocab_index[w] = i
        
    tail_vocab_index = {}
    for i, w in enumerate(tail_vocab):
        tail_vocab_index[w] = i

    # get the corpus of relation and tail    
    new_corpus = []
    for doc in all_doc:
        new_doc = []
        for word in doc:
            word_idx = vocab_index[word]
            new_doc.append(word_idx)
        new_corpus.append(new_doc)

    tail_corpus = []
    for doc in all_tail_doc:
        new_doc = []
        for word in doc:
            word_idx = tail_vocab_index[word]
            new_doc.append(word_idx)
        tail_corpus.append(new_doc)
    print(len(tail_corpus))
    print(len(tail_vocab))
    for i in range(len(new_corpus)):
        if len(new_corpus[i]) != len(tail_corpus[i]):
            print('ss')

    # get the all new triples facts 
    new_data_df = pd.DataFrame()
    for i in range(len(new_data)):
        temp = pd.DataFrame(new_data[i])
        new_data_df = new_data_df.append(temp,ignore_index=True)
    print(new_data_df)

    # get the all entities
    union_list = list(set().union(tail_vocab, list(reference_data.subject.values)))
    print(len(union_list))
    intersection_set = set.intersection(set(tail_vocab), set(list(reference_data.subject.values)))
    intersection_list = list(intersection_set)
    print(len(intersection_list))
    # print(data)

    return new_corpus, tail_corpus, vocab, tail_vocab