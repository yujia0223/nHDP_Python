from argparse import ArgumentParser
import pandas as pd
import json
from gensim import corpora
from coherence import *
from parent_child import *
from coverage import *


def evaluate_hierarchical_tree_notpath(tree_df, subject_documents, evaluation_path):
    """
    Evaluate the quality of a hierarchical topic tree based on the coherence and topic diversity of its nodes.

    Parameters:
    tree (dict): A hierarchical topic tree, where the keys are the levels of the tree and the values are lists of topics.
    documents (list): A list of documents, where each document is represented as a list of words.

    Returns:
    dict: A dictionary containing the coherence and topic diversity scores for the tree, as well as the BTQ, LTQ, and HTQ scores.
    """
    # Initialize lists to store the coherence and topic diversity scores for each level of the tree
    # tree_df = pd.read_csv(tree_path)
    # # remove the level0 NAN value
    # tree_df = tree_df.drop('level0',axis=1)
    # for col in tree_df.columns:
    #     tree_df[col] = tree_df[col].str.split(' ')

    # # read pickled documents
    # subject_documents = pd.read_pickle(documents_path)

    # texts = [[word for word in document.lower().split()]
    # for document in subject_documents]

    # dictionary = corpora.Dictionary(texts) # id2word
    # corpus = [dictionary.doc2bow(text) for text in texts]
    if isinstance(subject_documents, dict):
        texts = list(subject_documents.values())
    else:
        texts = subject_documents
    dictionary = corpora.Dictionary(texts)


    phi_WB, d_B = [], []
    phi_WL, d_L = [], []
    coverage_level_simple = []
    coverage_level_doc = []
    coverage_level_vocab = []

    # Iterate over the levels of the tree
    for level in tree_df.columns:
        # Convert each column to a list of tuples
        level_topics = [tuple(x) for x in tree_df[level].tolist()]
        level_topics = [topics for topics in level_topics if topics]
        print(f'level topics {level}', level_topics)
        # Get unique topics by converting the list to a set, then convert back to a list

        unique_topics = list(set(level_topics))

        # Print the unique topics
        print(f'unique topics in level {level}',unique_topics)

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


    # Compute coherence and topic diversity for each branch of the tree
    for index,row_branch in tree_df.iterrows():

        branch_topics = row_branch.tolist()
        print(f'branch topics {index}',branch_topics)

        # phi_WB.append(compute_coherence(branch_topics, corpus, dictionary))
        phi_WB.append(compute_coherence_cv(branch_topics, texts, dictionary))
        d_B.append(compute_topic_diversity(branch_topics))


    # Compute the BTQ and LTQ scores
    BTQ, LTQ = compute_btq_ltq(phi_WB, phi_WL, d_B, d_L)

    # Compute the HTQ score
    HTQ = compute_HTQ(BTQ, LTQ)

    # Compute the parent-child relatedness scores
    column_list = list(tree_df.columns)
    child_relatedness = {}
    non_child_relatedness = {}
    for level in column_list[1:]:

        print(f"Level: {level}")
        parent_index = column_list[column_list.index(level) - 1]
        parent_topics = tree_df[parent_index].drop_duplicates().tolist()
        print('parent topics', parent_topics)
        print(len(parent_topics))


        # if len(parent_topics) <= 1: # if need?
        #     child_topics = tree_df[level].drop_duplicates().tolist()
        #     print('child topics',child_topics)
        # else:

        all_topics = tree_df[level].drop_duplicates().tolist() # all topics in the level


        for p_topic in parent_topics:
            print(p_topic)
            child_topics = tree_df[tree_df[column_list[column_list.index(level) - 1]].apply(lambda x: x == p_topic)][level].drop_duplicates().tolist()
            print('child topics', child_topics)
            non_child_topics = [topic for topic in all_topics if topic not in child_topics]

            print('non child topics',non_child_topics)

            child_relatedness_t = calculate_relatedness(p_topic, child_topics, texts)
            child_relatedness[tuple(p_topic)] = child_relatedness_t
            print('child relatedness',child_relatedness_t)

            if non_child_topics: # if non child topics exist
                non_child_relatedness_t = calculate_relatedness(p_topic, non_child_topics, texts)
            # non_child_relatedness_t = calculate_relatedness(p_topic, non_child_topics, texts)
                non_child_relatedness[tuple(p_topic)] = non_child_relatedness_t
                print('non child relatedness',non_child_relatedness_t)
        # unique_topics = tree_df[levels].drop_duplicates().tolist()
        # print(f'topics in level {level}',unique_topics)

        # # Compute coherence and topic diversity for each list of unique topics
        # relate_score = calculate_relatedness(unique_topics, texts)
        # parent_child_relatedness.append(relate_score)

    # Return the coherence and topic diversity scores, as well as the BTQ, LTQ, and HTQ scores
    # dump the results to a file
    results = {
        'BTQ': BTQ,
        'LTQ': LTQ,
        'HTQ': HTQ,
        'Coverage_simple': coverage_level_simple,
        'Coverage_doc': coverage_level_doc,
        'Coverage_vocab':coverage_level_vocab,
        'child_relatedness': child_relatedness,
        'non_child_relatedness': non_child_relatedness,
    }
    # Convert tuples in keys to strings for JSON serialization
    def convert_keys_to_string(dictionary):
        return {str(key): value for key, value in dictionary.items()}
    # Apply conversion
    for key in ['Coverage_simple', 'child_relatedness', 'non_child_relatedness']:
        if isinstance(results[key], list):
            for i, entry in enumerate(results[key]):
                results[key][i] = convert_keys_to_string(entry)
        else:
            results[key] = convert_keys_to_string(results[key])
    print(results)
    with open(evaluation_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results

def evaluate_hierarchical_tree(tree_path, documents_path, evaluation_path):
    """
    Evaluate the quality of a hierarchical topic tree based on the coherence and topic diversity of its nodes.

    Parameters:
    tree (dict): A hierarchical topic tree, where the keys are the levels of the tree and the values are lists of topics.
    documents (list): A list of documents, where each document is represented as a list of words.

    Returns:
    dict: A dictionary containing the coherence and topic diversity scores for the tree, as well as the BTQ, LTQ, and HTQ scores.
    """
    # Initialize lists to store the coherence and topic diversity scores for each level of the tree
    tree_df = pd.read_csv(tree_path)
    # remove the level0 NAN value
    tree_df = tree_df.drop('level0',axis=1)
    for col in tree_df.columns:
        tree_df[col] = tree_df[col].str.split(' ')

    # read pickled documents
    subject_documents = pd.read_pickle(documents_path)

    # texts = [[word for word in document.lower().split()]
    # for document in subject_documents]

    # dictionary = corpora.Dictionary(texts) # id2word
    # corpus = [dictionary.doc2bow(text) for text in texts]
    texts = list(subject_documents.values())
    dictionary = corpora.Dictionary(texts)


    phi_WB, d_B = [], []
    phi_WL, d_L = [], []
    coverage_level_simple = []
    coverage_level_doc = []
    coverage_level_vocab = []

    # Iterate over the levels of the tree
    for level in tree_df.columns:
        # Convert each column to a list of tuples
        level_topics = [tuple(x) for x in tree_df[level].tolist()]
        level_topics = [topics for topics in level_topics if topics]
        print(f'level topics {level}', level_topics)
        # Get unique topics by converting the list to a set, then convert back to a list

        unique_topics = list(set(level_topics))

        # Print the unique topics
        print(f'unique topics in level {level}',unique_topics)

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


    # Compute coherence and topic diversity for each branch of the tree
    for index,row_branch in tree_df.iterrows():

        branch_topics = row_branch.tolist()
        print(f'branch topics {index}',branch_topics)

        # phi_WB.append(compute_coherence(branch_topics, corpus, dictionary))
        phi_WB.append(compute_coherence_cv(branch_topics, texts, dictionary))
        d_B.append(compute_topic_diversity(branch_topics))


    # Compute the BTQ and LTQ scores
    BTQ, LTQ = compute_btq_ltq(phi_WB, phi_WL, d_B, d_L)

    # Compute the HTQ score
    HTQ = compute_HTQ(BTQ, LTQ)

    # Compute the parent-child relatedness scores
    column_list = list(tree_df.columns)
    child_relatedness = {}
    non_child_relatedness = {}
    for level in column_list[1:]:

        print(f"Level: {level}")
        parent_index = column_list[column_list.index(level) - 1]
        parent_topics = tree_df[parent_index].drop_duplicates().tolist()
        print('parent topics', parent_topics)
        print(len(parent_topics))


        # if len(parent_topics) <= 1: # if need?
        #     child_topics = tree_df[level].drop_duplicates().tolist()
        #     print('child topics',child_topics)
        # else:

        all_topics = tree_df[level].drop_duplicates().tolist() # all topics in the level


        for p_topic in parent_topics:
            print(p_topic)
            child_topics = tree_df[tree_df[column_list[column_list.index(level) - 1]].apply(lambda x: x == p_topic)][level].drop_duplicates().tolist()
            print('child topics', child_topics)
            non_child_topics = [topic for topic in all_topics if topic not in child_topics]

            print('non child topics',non_child_topics)

            child_relatedness_t = calculate_relatedness(p_topic, child_topics, texts)
            child_relatedness[tuple(p_topic)] = child_relatedness_t
            print('child relatedness',child_relatedness_t)

            if non_child_topics: # if non child topics exist
                non_child_relatedness_t = calculate_relatedness(p_topic, non_child_topics, texts)
            # non_child_relatedness_t = calculate_relatedness(p_topic, non_child_topics, texts)
                non_child_relatedness[tuple(p_topic)] = non_child_relatedness_t
                print('non child relatedness',non_child_relatedness_t)
        # unique_topics = tree_df[levels].drop_duplicates().tolist()
        # print(f'topics in level {level}',unique_topics)

        # # Compute coherence and topic diversity for each list of unique topics
        # relate_score = calculate_relatedness(unique_topics, texts)
        # parent_child_relatedness.append(relate_score)

    # Return the coherence and topic diversity scores, as well as the BTQ, LTQ, and HTQ scores
    # dump the results to a file
    results = {
        'BTQ': BTQ,
        'LTQ': LTQ,
        'HTQ': HTQ,
        'Coverage_simple': coverage_level_simple,
        'Coverage_doc': coverage_level_doc,
        'Coverage_vocab':coverage_level_vocab,
        'child_relatedness': child_relatedness,
        'non_child_relatedness': non_child_relatedness,
    }
    # Convert tuples in keys to strings for JSON serialization
    def convert_keys_to_string(dictionary):
        return {str(key): value for key, value in dictionary.items()}
    # Apply conversion
    for key in ['Coverage_simple', 'child_relatedness', 'non_child_relatedness']:
        if isinstance(results[key], list):
            for i, entry in enumerate(results[key]):
                results[key][i] = convert_keys_to_string(entry)
        else:
            results[key] = convert_keys_to_string(results[key])
    print(results)
    with open(evaluation_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results

def main():
    parser = ArgumentParser(description='Evaluate hierarchical topics from a CSV tree and document corpus.')
    parser.add_argument('tree_csv_path', help='Path to the CSV file containing the hierarchical tree')
    parser.add_argument('documents_path', help='Path to the pickle file containing documents')
    parser.add_argument('evaluation_path', help='Path to save the evaluation results in JSON format')
    args = parser.parse_args()
    evaluate_hierarchical_tree(args.tree_csv_path, args.documents_path, args.evaluation_path)

if __name__ == '__main__':
    main()