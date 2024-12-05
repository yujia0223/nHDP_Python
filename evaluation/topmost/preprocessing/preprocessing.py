'''
This script is partially based on https://github.com/dallascard/scholar.
'''

import os
import re
import string
import gensim.downloader
from collections import Counter
import numpy as np
import scipy.sparse
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from topmost.data import file_utils


# Function to get the embeddings
def get_embedding(model, entity, embedding_type='e'):
    try:
        return model.get_embeddings([entity], embedding_type=embedding_type)[0]
    except KeyError:
        print(f"Entity or relation '{entity}' not found in the model. Returning zero vector.")
        return np.zeros(model.k)

def make_graph_embeddings(vocab, predicates, objects, model):
    # Initialize an array for storing combined embeddings
    embedding_dim = model.k  # Dimensionality of the KG embeddings
    combined_embeddings = np.zeros((len(vocab), embedding_dim))

    num_found = 0
    for i, word in enumerate(tqdm(vocab, desc="===> Making KG embeddings with KG model")):
        # print(f'===> Processing word: {word}')
        # print(f'===> predicate: {predicates}')
        # print(f'===> object: {objects}')
        if word in predicates:
        # Get KG embeddings for subject, relation, and object
            # subject_embedding = get_embedding(model, subject, embedding_type='e')
            relation_embedding = get_embedding(model, word, embedding_type='r')
            word_embedding = relation_embedding
        if word in objects:
            object_embedding = get_embedding(model, word, embedding_type='e')
            word_embedding = object_embedding
        

        combined_embeddings[i] = word_embedding
        num_found += 1

    print(f'===> Number of found embeddings: {num_found}/{len(vocab)}')

    return scipy.sparse.csr_matrix(combined_embeddings)

class Preprocessing:
    def __init__(self, tokenizer=None, test_sample_size=None, test_p=0.2, stopwords=None, min_doc_count=0, max_doc_freq=1.0, keep_num=False, keep_alphanum=False, strip_html=False, no_lower=False, min_length=3, min_term=1, vocab_size=None, seed=42):
        """
        Args:
            test_sample_size:
                Size of the test set.
            test_p:
                Proportion of the test set. This helps sample the train set based on the size of the test set.
            stopwords:
                List of stopwords to exclude [None|mallet|snowball].
            min-doc-count:
                Exclude words that occur in less than this number of documents.
            max_doc_freq:
                Exclude words that occur in more than this proportion of documents.
            keep-num:
                Keep tokens made of only numbers.
            keep-alphanum:
                Keep tokens made of a mixture of letters and numbers.
            strip_html:
                Strip HTML tags.
            no-lower:
                Do not lowercase text
            min_length:
                Minimum token length.
            min_term:
                Minimum term number
            vocab-size:
                Size of the vocabulary (by most common in the union of train and test sets, following above exclusions)
            seed:
                Random integer seed (only relevant for choosing test set)
        """

        self.test_sample_size = test_sample_size
        self.min_doc_count = min_doc_count
        self.max_doc_freq = max_doc_freq
        self.min_term = min_term
        self.test_p = test_p
        self.vocab_size = vocab_size
        self.seed = seed

    def parse(self, texts, vocab):
        if not isinstance(texts, list):
            texts = [texts]

        parsed_texts = list()
        for i, text in enumerate(tqdm(texts, desc="===>parse texts")):
            tokens = text.split()
            tokens = [t for t in tokens if t in vocab]
            parsed_texts.append(' '.join(tokens))

        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
        bow = vectorizer.fit_transform(parsed_texts)
        bow = bow.toarray()
        return parsed_texts, bow

    def preprocess_jsonlist(self, dataset_dir, label_name=None):
        train_items = file_utils.read_jsonlist(os.path.join(dataset_dir, 'train.jsonlist'))
        test_items = file_utils.read_jsonlist(os.path.join(dataset_dir, 'test.jsonlist'))

        print(f"Found training documents {len(train_items)} testing documents {len(test_items)}")

        raw_train_texts = []
        train_labels = []
        raw_test_texts = []
        test_labels = []

        for item in train_items:
            raw_train_texts.append(item['text'])

            if label_name is not None:
                train_labels.append(item[label_name])
 
        for item in test_items:
            raw_test_texts.append(item['text'])

            if label_name is not None:
                test_labels.append(item[label_name])

        rst = self.preprocess(raw_train_texts, train_labels, raw_test_texts, test_labels)

        return rst

    def convert_labels(self, train_labels, test_labels):
        if train_labels is not None:
            label_list = list(set(train_labels))
            label_list.sort()
            n_labels = len(label_list)
            label2id = dict(zip(label_list, range(n_labels)))

            print("label2id: ", label2id)

            train_labels = [label2id[label] for label in train_labels]

            if test_labels is not None:
                test_labels = [label2id[label] for label in test_labels]

        return train_labels, test_labels

    def preprocess(self, raw_train_texts, predicates, objects, embedding_model, train_labels=None, raw_test_texts=None, test_labels=None):
        np.random.seed(self.seed)

        train_texts = list()
        test_texts = list()
        word_counts = Counter()
        doc_counts_counter = Counter()

        train_labels, test_labels = self.convert_labels(train_labels, test_labels)

        for text in tqdm(raw_train_texts, desc="===>parse train texts"):
            tokens = text.split()
            word_counts.update(tokens)
            doc_counts_counter.update(set(tokens))
            parsed_text = ' '.join(tokens)
            train_texts.append(parsed_text)

        if raw_test_texts:
            for text in tqdm(raw_test_texts, desc="===>parse test texts"):
                tokens = text.split()
                word_counts.update(tokens)
                doc_counts_counter.update(set(tokens))
                parsed_text = ' '.join(tokens)
                test_texts.append(parsed_text)

        words, doc_counts = zip(*doc_counts_counter.most_common())
        doc_freqs = np.array(doc_counts) / float(len(train_texts) + len(test_texts))

        vocab = [word for i, word in enumerate(words) if doc_counts[i] >= self.min_doc_count and doc_freqs[i] <= self.max_doc_freq]

        # filter vocabulary
        if self.vocab_size is not None:
            vocab = vocab[:self.vocab_size]

        vocab.sort()

        train_idx = [i for i, text in enumerate(train_texts) if len(text.split()) >= self.min_term]
        train_idx = np.asarray(train_idx)

        if raw_test_texts is not None:
            test_idx = [i for i, text in enumerate(test_texts) if len(text.split()) >= self.min_term]
            test_idx = np.asarray(test_idx)

        # randomly sample
        if self.test_sample_size:
            print("===>sample train and test sets...")

            train_num = len(train_idx)
            test_num = len(test_idx)
            test_sample_size = min(test_num, self.test_sample_size)
            train_sample_size = int((test_sample_size / self.test_p) * (1 - self.test_p))
            if train_sample_size > train_num:
                test_sample_size = int((train_num / (1 - self.test_p)) * self.test_p)
                train_sample_size = train_num

            train_idx = train_idx[np.sort(np.random.choice(train_num, train_sample_size, replace=False))]
            test_idx = test_idx[np.sort(np.random.choice(test_num, test_sample_size, replace=False))]

            print(f"===>sampled train size: {len(train_idx)}")
            print(f"===>sampled train size: {len(test_idx)}")

        train_texts, train_bow = self.parse(np.asarray(train_texts)[train_idx].tolist(), vocab)

        rst = {
            'vocab': vocab,
            'train_bow': train_bow,
            'train_texts': train_texts,
            'word_embeddings': make_graph_embeddings(vocab, predicates, objects, embedding_model)
        }

        if train_labels is not None:
            rst['train_labels'] = np.asarray(train_labels)[train_idx]

        print(f"Real vocab size: {len(vocab)}")
        print(f"Real training size: {len(train_texts)} \t avg length: {rst['train_bow'].sum() / len(train_texts):.3f}")

        if raw_test_texts is not None:
            rst['test_texts'], rst['test_bow'] = self.parse(np.asarray(test_texts)[test_idx].tolist(), vocab)

            if test_labels is not None:
                rst['test_labels'] = np.asarray(test_labels)[test_idx]

            print(f"Real testing size: {len(rst['test_texts'])} \t avg length: {rst['test_bow'].sum() / len(rst['test_texts']):.3f}")

        return rst

    def save(self, output_dir, vocab, train_texts, train_bow, word_embeddings, train_labels=None, test_texts=None, test_bow=None, test_labels=None):
        file_utils.make_dir(output_dir)

        file_utils.save_text(vocab, f"{output_dir}/vocab.txt")
        file_utils.save_text(train_texts, f"{output_dir}/train_texts.txt")
        scipy.sparse.save_npz(f"{output_dir}/train_bow.npz", scipy.sparse.csr_matrix(train_bow))
        scipy.sparse.save_npz(f"{output_dir}/word_embeddings.npz", word_embeddings)

        if train_labels is not None:
            np.savetxt(f"{output_dir}/train_labels.txt", train_labels, fmt='%i')

        if test_bow is not None:
            scipy.sparse.save_npz(f"{output_dir}/test_bow.npz", scipy.sparse.csr_matrix(test_bow))

        if test_texts is not None:
            file_utils.save_text(test_texts, f"{output_dir}/test_texts.txt")

            if test_labels is not None:
                np.savetxt(f"{output_dir}/test_labels.txt", test_labels, fmt='%i')
