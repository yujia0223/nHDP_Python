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


# Step 1: Extract the tar.gz file
tar_path = 'data/20news-bydate.tar.gz'
extract_path = 'data/20news-bydate'

# with tarfile.open(tar_path, "r:gz") as tar:
#     tar.extractall(path=extract_path)

# Step 2: Load the dataset
def load_20ng_dataset(directory):
    categories = os.listdir(directory)
    data = []
    for category in categories:
        category_path = os.path.join(directory, category)
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    content = f.read()
                    data.append({'category': category, 'filename': filename, 'content': content})
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return pd.DataFrame(data)

# Assuming the dataset is divided into training and testing sets
train_path = os.path.join(extract_path, '20news-bydate-train')
test_path = os.path.join(extract_path, '20news-bydate-test')

train_df = load_20ng_dataset(train_path)
test_df = load_20ng_dataset(test_path)

print(train_df)
print(test_df)


# Step 3: Preprocess the dataset
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if len(token) > 3]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Apply the preprocessing function to the 'content' column in train_df
train_df['content'] = train_df['content'].apply(preprocess_text)

print(train_df)

# # Convert the preprocessed text in train_df to a list of documents
# documents = train_df['content'].tolist()

# # Create a dictionary mapping each word to a unique integer ID
# dictionary = corpora.Dictionary(documents)

# # Create a corpus representation of the documents
# corpus = [dictionary.doc2bow(doc) for doc in documents]

# # Get the vocabulary of the corpus
# vocab = list(dictionary.values())

corpus = []
all_docs = []
vocab = set()
for i in range(len(train_df)):
    doc = train_df.iloc[i]['content']
    all_docs.append(doc)
    tokens = doc.split()
    corpus.append(tokens)
    vocab.update(tokens)

# Print the corpus and vocabulary
# print(corpus)
# print(vocab)
vocab = sorted(list(vocab))
vocab_index = {}
for i, w in enumerate(vocab):
    vocab_index[w] = i

new_corpus = []
for doc in corpus:
    new_doc = []
    for word in doc:
        word_idx = vocab_index[word]
        new_doc.append(word_idx)
    new_corpus.append(new_doc)

print(len(vocab), len(corpus), len(corpus[0]), len(corpus[1]))
print(len(vocab), len(new_corpus), len(new_corpus[0]), len(new_corpus[1]))
print(corpus[0][0:10])
print(new_corpus[0][0:10])

# Visualize the word cloud of the entire corpus
wordcloud = WordCloud(background_color='white').generate(' '.join(all_docs))
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

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
save_zipped_pickle(hlda, f'saved_models/{timestamp}_20ng_hlda.p')


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
paths_df = pd.DataFrame(paths)

# Step 7: Evaluate the hLDA results
texts = corpus
dictionary = corpora.Dictionary(texts)

phi_WB, d_B = [], []
phi_WL, d_L = [], []
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

BTQ, LTQ = compute_btq_ltq(phi_WB, phi_WL, d_B, d_L)
HTQ = compute_HTQ(BTQ, LTQ)
print('BTQ_scores:', BTQ)
print('LTQ_scores:', LTQ)
print('HTQ:', HTQ)