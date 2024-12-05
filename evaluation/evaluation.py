import csv
from sklearn.metrics import jaccard_score
import glob
import pandas as pd
import numpy as np
import operator
from io import StringIO

def load_tree(tree_csv_path):
    csv.field_size_limit(int(2**31 - 1))
    with open(tree_csv_path) as f:
        nodes = dict(
            (
                tuple(int(x) for x in row['me'].strip().split()),
                dict(parent_loc=tuple(int(x) for x in row['parent'].strip().split()),
                     tau_sums=float(row['tau_sums'].strip()),
                     lambda_sums=[float(x) for x in row['lambda_sums'].strip().split()],
                     children={})
            )
            for row in csv.DictReader(f)
        )

    root_loc = (1,)
    if root_loc not in nodes:
        nodes[root_loc] = dict(parent_loc=(), children={})

    for (node_loc, node) in nodes.items():
        parent_loc = node['parent_loc']
        if parent_loc:
            child_idx = node_loc[-1]
            nodes[parent_loc]['children'][child_idx] = node
    
    leaves_topics = []
    for (node_loc, node) in nodes.items():
        node['me'] = node_loc
        for (_, child) in sorted(node['children'].items()):
            node['children'] = [child]
        # node['children'] = [child for (_, child) in sorted(node['children'].items())]
        if not node['children']:
            leaves_topics.append(node['me'])


    return nodes[root_loc], leaves_topics

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

# load the raw data from directory
raw_data = pd.read_csv('data/dbpedia/triples.txt', delimiter = '\t')
raw_data = raw_data.to_numpy()
print('---------- load the data ----------')

# load the labels for evaluation
reference_data = pd.read_csv('data/dbpedia/classes.txt', delimiter = '\t')
# reference_data['class_int'] = pd.Categorical(reference_data['level2']).codes
reference_data['newlabel'] = reference_data['level2'].str.get_dummies().values.tolist()
subjects = set(raw_data[:,0])
# check if 
for i in range(len(reference_data)):
    tmpt = reference_data.subject[i]
    if tmpt not in subjects:
        reference_data = reference_data.drop(i)



# predict results
# read the subtree for each document
sum_score = []
docs = '../nHDP_matlab/output/tree/dbpedia/nhdp_tree_docs_443.csv' # remember to change

# # Convert String into StringIO
# csvStringIO = StringIO(docs)
Tree_docs_df = pd.read_csv(docs, dtype=str)
# docs_path = '../nHDP_matlab/output/subtree/dbpedia/*.csv' # TODO: change it test
docs_path = '../nHDP_matlab/output/testing/*.csv' # TODO: change it test

# All files and directories ending with .txt and that don't begin with a dot:
docs_subtree = glob.glob(docs_path)
# # All files and directories ending with .txt with depth of 2 folders, ignoring names beginning with a dot:
# print(glob.glob("/home/adam/*/*.txt"))

# print(len(docs_subtree)).
label_list = sorted(list(set(reference_data.level2)))
for doc in range(len(docs_subtree)):
    print(doc)
    subtree_nodes, leaves_topics = load_tree(docs_subtree[doc])
    # print(subtree_nodes)
    print('leaves of subtree',leaves_topics)

    # choose the leaves of subtree
    # how to find if is leaves:
    # to check if this node has children
    y_pred = []
    y_true = []
    for topic in leaves_topics:

        # first label the cluster as per the real label of the documents in this cluster
        node_me = str(topic).replace(',','').replace('(','').replace(')','')
        docs_incluster = Tree_docs_df[Tree_docs_df.me == node_me]['docs'].values[0].split(' ') # TODO: confirm the evaluation method
        # print(docs_incluster)
        docs_incluster_list = [np.int(i) for i in docs_incluster]
        true_label = reference_data.level2[list(docs_incluster_list)]
        counts = word_count_func(true_label) # confirm
        # index_label = np.argmax(counts.values)
        label_assign = max(counts.items(), key=operator.itemgetter(1))[0]
        y_pred.append(label_assign)
        y_true.append(reference_data.level2[doc])
    # evaluate it on the leaves of the subtree
    # for each document compare real label and clustering label
    print(y_true)
    print(y_pred)
    score_macro = jaccard_score(y_true, y_pred, average='macro')
    score_micro = jaccard_score(y_true, y_pred, average='micro')
    sum_score.append([score_macro, score_micro]) 

avg_score = np.mean(sum_score, axis = 0) # ?how to have all multi label score?
print(avg_score)