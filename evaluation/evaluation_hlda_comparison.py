import tarfile
import os
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from hlda.sampler import HierarchicalLDA
from wordcloud import WordCloud
# import cPickle
import gzip
# from ipywidgets import widgets
# from IPython.core.display import HTML, display
import pylab as plt
import datetime
import csv
import dill
import gzip
from coherence import *
from parent_child import *
import time

# pip install nltk
# pip install hlda
# pip install wordcloud


def preprocess(raw_data, reference_data):
    """
    Preprocess raw data by organizing entities, relations, and objects based on a reference dataset.

    Args:
    raw_data (pd.DataFrame): DataFrame of the raw data containing columns for subject, predicate, and object.
    reference_data (pd.DataFrame): DataFrame containing entity references.

    Returns:
    tuple: A tuple containing:
        - subject_documents (dict): Dictionary with subjects as keys and lists of predicates and objects as values.
        - vocab (list): Sorted list of the vocabulary.
        - new_data_df (pd.DataFrame): DataFrame containing processed triples.
        - all_subjects (list): List of all subjects found in the reference data.
    """

    # Filter raw_data to include subjects found in reference_data
    filtered_data = raw_data[raw_data['Subject'].isin(reference_data['entities'])]

    # Extracting subjects
    all_subjects = filtered_data['Subject'].unique().tolist()
    print(f"Number of subjects: {len(all_subjects)}")

    # Grouping predicates and objects by subjects
    grouped = filtered_data.groupby('Subject')
    # print('Grouping predicates and objects by subjects',grouped)

    subject_documents = {subj: group[['Predicate', 'Object']].values.flatten().tolist() for subj, group in grouped}
    print('subject_documents',subject_documents)

    # Creating vocabulary
    vocab = sorted(set(sum(subject_documents.values(), [])))  # Flattens the list of lists and then converts to a set for unique elements
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # Creating a corpus with indexed vocabulary
    indexed_corpus = {subj: [vocab_index[word] for word in words] for subj, words in subject_documents.items()}

    return indexed_corpus, vocab, filtered_data, all_subjects, subject_documents

# Load the raw data as a DataFrame

raw_data = pd.read_csv('data/fb15k-237/train.txt', delimiter='\t', names=['Subject', 'Predicate', 'Object'])
print('---------- Raw data loaded successfully ----------')

# Load reference data and filter out unreferenced entities
# reference_data = pd.read_csv('data/fb15k-237/freebase_reference_data_nhdp_20231030-154020.csv')
# reference_data = pd.read_csv('data/fb15k-237/freebase_reference_data_nhdp_20231108-232426.csv')
reference_data = pd.read_csv('data/fb15k-237/freebase_reference_data_nhdp_20231207-075638.csv')
# reference_data = pd.read_csv('data/fb15k-237/freebase_reference_data_nhdp_20231215-160315.csv')
# reference_data = pd.read_csv('data/fb15k-237/freebase_reference_data_nhdp_20240102-162052.csv')
# reference_data = pd.read_csv('data/fb15k-237/freebase_reference_data_nhdp_20240102-170616.csv')
reference_data = reference_data[reference_data['entities'].isin(raw_data['Subject'])].reset_index(drop=True)
print(len(set(reference_data.classes)))
print([i for i in set(reference_data.classes)])

# Preprocessing
indexed_corpus, vocab, new_data_df, all_subjects, subject_documents = preprocess(raw_data, reference_data)

# Output sample for verification
print(f"Sample processed data:\n{new_data_df}")
print(f"Sample indexed corpus for a subject:\n{next(iter(indexed_corpus.items()))}")
print(f"Sample vocabulary:\n{vocab[:10]}")

new_corpus = indexed_corpus.values()


start_time = time.time()

# Step 4: Train an hLDA model

n_samples = 200       # no of iterations for the sampler
alpha = 10.0          # smoothing over level distributions
gamma = 1.0           # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
eta = 0.1             # smoothing over topic-word distributions
num_levels = 3        # the number of levels in the tree
display_topics = 50   # the number of iterations between printing a brief summary of the topics so far
n_words = 5           # the number of most probable words to print for each topic after model estimation
with_weights = False  # whether to print the words with the weights

hlda = HierarchicalLDA(new_corpus, vocab, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)

hlda_time = time.time() - start_time
print("Time taken for training hLDA model:", hlda_time)

start_time = time.time()

hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)

estimate_time = time.time() - start_time
print("Time taken for estimating hLDA model:", estimate_time)

# # Step 5: Visualize the topics
# colour_map = {
#     0: 'blue',
#     1: 'red',
#     2: 'green'
# }

# def show_doc(d=0):

#     node = hlda.document_leaves[d]
#     path = []
#     while node is not None:
#         path.append(node)
#         node = node.parent
#     path.reverse()

#     n_words = 10
#     with_weights = False
#     for n in range(len(path)):
#         node = path[n]
#         colour = colour_map[n]
#         msg = 'Level %d Topic %d: ' % (node.level, node.node_id)
#         msg += node.get_top_words(n_words, with_weights)
#         output = '<h%d><span style="color:%s">%s</span></h3>' % (n+1, colour, msg)
#         display(HTML(output))

#     display(HTML('<hr/><h5>Processed Document</h5>'))

#     doc = corpus[d]
#     output = ''
#     for n in range(len(doc)):
#         w = doc[n]
#         l = hlda.levels[d][n]
#         colour = colour_map[l]
#         output += '<span style="color:%s">%s</span> ' % (colour, w)
#     display(HTML(output))

# widgets.interact(show_doc, d=(0, len(corpus)-1))


def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        dill.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_obj = dill.load(f)
    return loaded_obj
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")    
save_zipped_pickle(hlda, f'comparison/saved_models/{timestamp}_fb_hlda.p')


# Step 6: convert the hLDA model tree to a pandas DataFrame
# List to hold node information
node_list = []

def save_nodes_to_df(hlda, n_words, with_weights):
    global node_list  # Make sure we are using the global list
    node_list = []  # Reset the list for each call
    save_node_to_df(hlda.root_node, None, 0, n_words, with_weights)  # Start with no parent for the root
    return pd.DataFrame(node_list)

def save_node_to_df(node, parent_id, indent, n_words, with_weights):
    string_list = node.get_top_words(n_words, with_weights)
    # print(string_list)
    converted_list = [string for string in string_list.split(',')[:-1]]  # This splits each string into a list based on commas
    node_info = {
        'node_id': node.node_id,  # Assuming each node has a unique identifier
        'parent_id': parent_id,
        'level': node.level,
        'customers': node.customers,
        'top_words': converted_list,
    }
    node_list.append(node_info)
    
    # Recursively save children nodes
    for child in node.children:
        save_node_to_df(child, node.node_id, indent + 1, n_words, with_weights)

# Example usage:
tree_df = save_nodes_to_df(hlda, n_words=5, with_weights=False)
tree_df.to_csv(f'comparison/saved_models/{timestamp}_fb_hlda.csv')
print(tree_df)

# Convert parent_id to int where possible, keeping NA for root nodes
tree_df['parent_id'] = pd.to_numeric(tree_df['parent_id'], errors='coerce')

# Function to find the path to a given node
def find_path(node_id, tree_df):
    path = []
    current_id = node_id
    while True:
        row = tree_df.loc[tree_df['node_id'] == current_id].iloc[0]
        path.insert(0, row['top_words'])  # Prepend to path
        if pd.isna(row['parent_id']):  # Reached the root
            break
        current_id = row['parent_id']
    return path

# Extract paths for all leaf nodes (nodes without children)
leaf_nodes = set(tree_df['node_id']) - set(tree_df['parent_id'].dropna())
paths = {node: find_path(node, tree_df) for node in leaf_nodes}
paths_df = pd.DataFrame(paths).T

# Step 7: Evaluate the hLDA results
texts = list(subject_documents.values())
dictionary = corpora.Dictionary(texts)

phi_WB, d_B = [], []
phi_WL, d_L = [], []
coverage_level_simple = []
coverage_level_doc = []
coverage_level_vocab = []

for leaf_node in paths_df.columns:
    print(leaf_node)
    branch_topics = paths_df[leaf_node].tolist()
    print(f'branch topics {leaf_node}',branch_topics)

    # phi_WB.append(compute_coherence(branch_topics, corpus, dictionary))
    phi_WB.append(compute_coherence_cv(branch_topics, texts, dictionary))
    d_B.append(compute_topic_diversity(branch_topics))


for level_index in set(tree_df.level):
    # Convert each column to a list of tuples
    unique_topics = tree_df[tree_df['level']== level_index]['top_words'].tolist()
    
    # Get unique topics by converting the list to a set, then convert back to a list
    # unique_topics = list(set(topics))
    
    # Print the unique topics
    print(f'unique topics in level {level_index}',unique_topics)
    
    # Compute coherence and topic diversity for each list of unique topics
    # phi_WL.append(compute_coherence(unique_topics, corpus, dictionary))
    phi_WL.append(compute_coherence_cv(unique_topics, texts, dictionary))
    d_L.append(compute_topic_diversity(unique_topics))

    # Compute coherence and topic diversity for each list of unique topics
    coverage_score_simple = calculate_coverage_simple(unique_topics, texts)
    # coverage_score = calculate_coverage(unique_topics, texts)
    coverage_score_doc = calculate_coverage_doc(unique_topics, texts)
    coverage_score_vocab = calculate_coverage_vocab(unique_topics, texts)

    # coverage_level.append(coverage_score)
    coverage_level_simple.append(coverage_score_simple)
    coverage_level_doc.append(coverage_score_doc)
    coverage_level_vocab.append(coverage_score_vocab)

BTQ, LTQ = compute_btq_ltq(phi_WB, phi_WL, d_B, d_L)
HTQ = compute_HTQ(BTQ, LTQ)
print('BTQ_scores:', BTQ)
print('LTQ_scores:', LTQ)
print('HTQ:', HTQ)

