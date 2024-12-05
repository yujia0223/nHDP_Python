import pandas as pd
from coherence import *
from parent_child import *
from coverage import *
from evaluation_scripts_range import evaluate_hierarchical_tree_notpath
import topmost
from topmost.preprocessing import Preprocessing
import json
import numpy as np
from topmost import evaluations
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.utils import save_model, restore_model
from tqdm import tqdm
import scipy.sparse
import torch

def preprocess(raw_data):
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
    filtered_data = raw_data

    # Extracting subjects
    all_subjects = filtered_data['Subject'].unique().tolist()
    print(f"Number of subjects: {len(all_subjects)}")

    # Grouping predicates and objects by subjects
    grouped = filtered_data.groupby('Subject')
    # print('Grouping predicates and objects by subjects',grouped)

    subject_documents = {subj: group[['Predicate', 'Object']].values.flatten().tolist() for subj, group in grouped}
    # print('subject_documents',subject_documents)

    # Creating vocabulary
    vocab = sorted(set(sum(subject_documents.values(), [])))  # Flattens the list of lists and then converts to a set for unique elements
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # Creating a corpus with indexed vocabulary
    indexed_corpus = {subj: [vocab_index[word] for word in words] for subj, words in subject_documents.items()}

    # Extracting predicates and objects
    predicates = filtered_data['Predicate'].unique().tolist()
    objects = filtered_data['Object'].unique().tolist()

    return indexed_corpus, vocab, filtered_data, all_subjects, subject_documents, predicates, objects

# Load the raw data as a DataFrame

raw_data = pd.read_csv('data/fb15k-237/train.txt', delimiter='\t', names=['Subject', 'Predicate', 'Object'])
print('---------- Raw data loaded successfully ----------')

# Preprocessing
indexed_corpus, vocab, new_data_df, all_subjects, subject_documents, predicates, objects = preprocess(raw_data)

# Output sample for verification
# print(f"Sample processed data:\n{new_data_df}")
# print(f"Sample indexed corpus for a subject:\n{next(iter(indexed_corpus.items()))}")
# print(f"Sample vocabulary:\n{vocab[:10]}")

def format_subject_documents(subject_documents):
    documents = []  # List to store each subject's document as a separate string
    for subject, words in subject_documents.items():
        document_string = ''
        # Concatenate words with commas and add a period at the end
        document_string += ' '.join(words) + '.'
        documents.append(document_string)
    return documents
# Call the function to create the document
docs = format_subject_documents(subject_documents)
# docs = list(indexed_corpus.values())
# print(docs)

# train the embedding of KG
# Train the TransE model
data_array = raw_data.to_numpy()
embedding_model = ScoringBasedEmbeddingModel(k=50, eta=1, scoring_type='TransE')
embedding_model.compile(optimizer='adam', loss='nll')
embedding_model.fit(data_array, epochs=100)
# Save the model
path_embedding = "embedding/fb15k-237.pkl"
save_model(embedding_model, model_name_path=path_embedding)

# Restore the model
# restored_model = restore_model(model_name_path=path_embedding)


# tokenized_docs = [doc.split() for doc in docs]
# # print('after token',tokenized_docs)
# device = 'cuda' # or 'cpu'
# preprocessing = Preprocessing()
# dataset = topmost.data.RawDatasetHandler(docs, preprocessing, predicates, objects, embedding_model, batch_size = 64, device=device, as_tensor=True)
# # print(dataset)
# # print('train texts', dataset.train_texts)

# # torch.cuda.empty_cache()

# # create a model
# # model = topmost.models.SawETM(vocab_size=dataset.vocab_size, num_topics_list=[10, 50, 200], device=device)
# # model = topmost.models.HyperMiner(vocab_size=dataset.vocab_size, num_topics_list=[10, 50, 200], device=device)
# model = topmost.models.TraCo(dataset.vocab_size, num_topics_list=[2,8],bias_topk=5)
# model = model.to(device)
# # create a trainer
# trainer = topmost.trainers.HierarchicalTrainer(model)

# # train the model
# trainer.train(dataset)

# # Or directly use fit_transform
# # top_words, train_theta = trainer.fit_transform(dataset)

# # evaluate quality of topic hierarchy
# beta_list = trainer.export_beta()
# phi_list = trainer.export_phi()
# annoated_top_words = trainer.export_top_words(dataset.vocab, annotation=True)
# print(annoated_top_words)
# reference_bow = dataset.train_data.detach().cpu().numpy()
# print(type(reference_bow))
# results, topic_hierarchy = evaluations.hierarchy_quality(dataset.vocab, reference_bow, annoated_top_words, beta_list, phi_list)

# hierarchy_str = json.dumps(topic_hierarchy, indent=4)
# print(json.dumps(topic_hierarchy, indent=4))
# print(results)

# def traverse_hierarchy(node, path, results):
#     if isinstance(node, dict):
#         for key, value in node.items():
#             traverse_hierarchy(value, path + [key], results)
#     elif isinstance(node, list):
#         for item in node:
#             results.append(path + [item])

# def hierarchy_to_dataframe(hierarchy):
#     results = []
#     traverse_hierarchy(hierarchy, [], results)
#     # Find the maximum depth to standardize DataFrame column size
#     max_depth = max(len(result) for result in results)
#     # Create DataFrame
#     df = pd.DataFrame(results, columns=[f'level{i}' for i in range(max_depth)])
#     return df

# tree_df = hierarchy_to_dataframe(topic_hierarchy)
# print(tree_df)
# for col in tree_df.columns:
#     tree_df[col] = tree_df[col].str.replace(r'L-\d+_K-\d+ ', '', regex=True)
#     tree_df[col] = tree_df[col].str.split(' ')
# print(tree_df)
# timestamp = '20240426_all'


# evaluate_hierarchical_tree_notpath(tree_df, subject_documents, f'evaluation_score/freebase/{timestamp}_traco_fb.json')