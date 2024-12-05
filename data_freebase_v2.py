import pandas as pd
from datetime import datetime

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
try:
    raw_data = pd.read_csv('data/fb15k-237/train.txt', delimiter='\t', names=['Subject', 'Predicate', 'Object'])
    print('---------- Raw data loaded successfully ----------')

    # Load reference data and filter out unreferenced entities
    reference_data = pd.read_csv('data/fb15k-237/freebase_reference_data_nhdp_20231030-154020.csv')
    reference_data = reference_data[reference_data['entities'].isin(raw_data['Subject'])].reset_index(drop=True)
    print(reference_data)

    # Preprocessing
    indexed_corpus, vocab, new_data_df, all_subjects, subject_documents = preprocess(raw_data, reference_data)

    # Output sample for verification
    print(f"Sample processed data:\n{new_data_df}")
    print(f"Sample indexed corpus for a subject:\n{next(iter(indexed_corpus.items()))}")
    print(f"Sample vocabulary:\n{vocab[:10]}")

except Exception as e:
    print(f"An error occurred: {e}")




def word_count_func(text):
    '''
    Counts words within a string.
    
    Args:
        text (str): String to be processed.
    
    Returns:
        dict: A dictionary with words as keys and counts as values.
    ''' 
    counts = {}
    words = text.split() if isinstance(text, str) else text

    for word in words:
        counts[word] = counts.get(word, 0) + 1

    return counts

# Replace 'new_corpus' with your actual corpus data
new_corpus = indexed_corpus  # Your list of documents goes here
vocab = vocab  # Your list of vocabulary words goes here
all_data = {'doc_idx': [], 'word_idx': [], 'count': []}

# Generate a timestamp
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

for doc_idx, (doc_key, text_value) in enumerate(subject_documents.items()):
    words_count_dict = word_count_func(text_value)
    for word, count in words_count_dict.items():
        all_data['doc_idx'].append(doc_idx)
        word_idx = vocab.index(word) if word in vocab else -1  # -1 if the word is not found
        all_data['word_idx'].append(word_idx)
        all_data['count'].append(count)

corpus_df = pd.DataFrame(all_data)

# Save the DataFrame to a CSV file
filename = f'data/fb15k-237/corpus_po_{timestamp}.txt'
corpus_df.to_csv(filename, sep=' ', index=False, header=False)
print(f'DataFrame saved to {filename}')

# Optional: Print statements (can be removed or commented out in production code)
print(len(all_data['doc_idx']))
print(corpus_df)
print(corpus_df[corpus_df['count'] > 1])  # Rows where count is greater than 1

# Create and save the vocabulary DataFrame
vocab_df = pd.DataFrame({'vocab': vocab})
vocab_filename = f'data/fb15k-237/vocab_po_{timestamp}.txt'
vocab_df.to_csv(vocab_filename, index=False, header=False)
print(f'Vocabulary DataFrame saved to {vocab_filename}')
print(vocab_df.iloc[210])
