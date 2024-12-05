import numpy as np
import pandas as pd
from datetime import datetime
import re
# from scipy.sparse import coo_matrix
import math

from sklearn.cluster import KMeans


def preprocess(raw_data, reference_data):
    # preprocessing to get the documents and set of relation and tail       
    relation_list = []

    all_docs = []
    vocab = set()
    
    new_data = []
    all_subjects = []
    # get the string info for each subject
    for sub_type_index in reference_data.index:
         if (raw_data[:,0] == reference_data.entities[sub_type_index]).any():
            s = reference_data.entities[sub_type_index]
            all_subjects.append(s)
            index = np.where(raw_data[:,0]==s)[0]
            new_doc = []
            new_data.append(raw_data[index])
        
            for i in index:
                new_doc.append(raw_data[i,1]) # just add predicate
                new_doc.append(raw_data[i,2]) # just add object
            # new_doc.append(reference_data.level1[sub_type_index])
            # new_doc.append(reference_data.level2[sub_type_index])
                
        
        #         print(raw_data[i,1])
        #         print(raw_data[i,0])
            all_docs.append(new_doc)
            vocab.update(new_doc)


    # for s in reference_data.entities:
    #     if (raw_data[:,0] == s).any():
    #         index = np.where(raw_data[:,0]==s)[0]
    #         new_doc = []
    #         new_data.append(raw_data[index])
        
    #         for i in index:
    #             new_doc.append(raw_data[i,1]) # just add predicate
    #             new_doc.append(reference_data)
    #             # new_doc.append(raw_data[i,2]) # just add object
        
    #     #         print(raw_data[i,1])
    #     #         print(raw_data[i,0])
    #         all_docs.append(new_doc)
    #         vocab.update(new_doc)

    vocab = sorted(list(vocab))

    print('vocabulary is : ', vocab[0:10])
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
    new_data_list = []
    for i in range(len(new_data)):
        # print(i)
        new_data_list.append(new_data[i])
        # temp = pd.DataFrame(new_data[i])
        # new_data_df = new_data_df.append(temp,ignore_index=True)
    # print(new_data_df)
    new_data_df = pd.DataFrame(new_data_list)

    return new_corpus, vocab, new_data_df, all_subjects

# load the raw data from directory
raw_data = pd.read_csv('data/fb15k-237/train.txt', delimiter = '\t')
raw_data = raw_data.to_numpy()
print('---------- load the data ----------')

# load the labels for evaluation
reference_data = pd.read_csv('data/fb15k-237/freebase_reference_data_nhdp_20231030-154020.csv')
subjects = set(raw_data[:,0])
unique_entities_total = reference_data['entities'].unique()
# check if 
for i in range(len(reference_data)):
    tmpt = reference_data.entities[i]
    if tmpt not in subjects:
        reference_data = reference_data.drop(i)

# preprocessing to get the all corpus and set of relation and tail
new_corpus, vocab, new_data_df, all_subjects = preprocess(raw_data, reference_data)
print('the new data is: ', new_data_df)
print('the number of 1st level categores:', len(set(reference_data.level1)))
print('the 1st level categores:',set(reference_data.level1))
print('the number of 2nd level categores:',len(set(reference_data.level2)))
print('reference data:',reference_data)
#timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
new_data_df.to_csv('data/fb15k-237/all_triples_extracted_po.csv', index=False, header = False)

# def word_count_func(text):
#     '''
#     Counts words within a string
    
#     Args:
#         text (str): String to which the function is to be applied, string
    
#     Returns:
#         Number of words within a string, integer
#     ''' 
#     counts = dict()
#     # words = text.split() #when str
#     words = text # when list

#     for word in words:
#         if word in counts:
#             counts[word] += 1
#         else:
#             counts[word] = 1

#     return counts


# all = {'doc_idx':[], 'word_idx':[], 'count':[]}

# for doc in range(len(new_corpus)):


#   words_count_dict = word_count_func(new_corpus[doc])

#   for word in words_count_dict.keys():
#     all['doc_idx'].append(doc)

#     # word_idx = vocab.index(word)
#     all['word_idx'].append(word)

#     all['count'].append(words_count_dict[word])

# corpus_df = pd.DataFrame(all)
# # t is for label

# corpus_df.to_csv(f'data/fb15k-237/corpus_ot_{timestamp}.txt', sep = ' ', index=False, header = False)
# print(len(all['doc_idx']))
# print(corpus_df)
# print(corpus_df[corpus_df['count']>1]) #2,3,4,5,6,7

# vocab_df = pd.DataFrame(vocab)
# vocab_df.to_csv(f'data/fb15k-237/vocab_ot_{timestamp}.txt', index=False, header = False)
# print(vocab_df)
# print(vocab_df.iloc[210])

# # to check the original data
# # new_data_df = pd.read_csv('data/fb15k-237/all_triples_extracted_po.txt', index=False, header = False)
# # all_subjects = list(set(reference_data.s))
# new_data_df.columns = ['s','p','o']
# print('the 1st document is: \n')
# print(new_data_df[new_data_df.s == all_subjects[0]])
# print(list(new_data_df[new_data_df.s == all_subjects[0]].p))
# print('the 2nd document is: \n')
# print(new_data_df[new_data_df.s == all_subjects[1]])
# print(list(new_data_df[new_data_df.s == all_subjects[1]].p))
# print('the 99th document is: \n')
# print(new_data_df[new_data_df.s == all_subjects[99]])
# print(list(new_data_df[new_data_df.s == all_subjects[99]].p))

# new_data_df.columns = ['s','p','o']
# print('the 1st document is: \n')
# print(new_data_df[new_data_df.s == all_subjects[0]])
# print(list(new_data_df[new_data_df.s == all_subjects[510]]))
# print('the 2nd document is: \n')
# print(new_data_df[new_data_df.s == all_subjects[1]])
# print(list(new_data_df[new_data_df.s == all_subjects[1050]]))
# print('the 99th document is: \n')
# print(new_data_df[new_data_df.s == all_subjects[99]])
# print(list(new_data_df[new_data_df.s == all_subjects[4335]]))