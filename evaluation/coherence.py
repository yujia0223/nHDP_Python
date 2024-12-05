import pandas as pd
import numpy as np
from gensim import corpora
# from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from gensim.models.coherencemodel import CoherenceModel


# Topic coherence
def compute_coherence(topics, corpus, dictionary):
    cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    # cm = CoherenceModel(topics=topics, dictionary=dictionary, coherence='c_v')
    topic_coherence = cm.get_coherence()
    return topic_coherence

def compute_coherence_cv(topics, texts, dictionary):
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
    topic_coherence = cm.get_coherence()
    return topic_coherence

def compute_coherence_cv_pmi(topics, texts, dictionary):
    '''  coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
            Coherence measure to be used.
            Fastest method - 'u_mass', 'c_uci' also known as `c_pmi`.
            For 'u_mass' corpus should be provided, if texts is provided, it will be converted to corpus
            using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' `texts` should be provided (`corpus` isn't needed)'''
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_uci')
    topic_coherence = cm.get_coherence()
    return topic_coherence

def compute_topic_diversity(topics, topk = 5):

    """
    Retrieves the score of the metric

    Parameters
    ----------
    model_output : dictionary, output of the model
                    key 'topics' required.

    Returns
    -------
    td : score
    """
    topics = topics
    if topics is None:
        return 0
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than ' + str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        td = len(unique_words) / (topk * len(topics))
        return td


# Compute BTQ and LTQ
def compute_btq_ltq(phi_WB, phi_WL, d_B, d_L):
    BTQ = np.sum([phi * d for phi, d in zip(phi_WB, d_B)])/len(phi_WB)
    LTQ = np.sum([phi * d for phi, d in zip(phi_WL, d_L)])/len(phi_WL)
    return BTQ, LTQ

# Compute HTQ
def compute_HTQ(BTQ, LTQ):
    return (BTQ + LTQ) / 2


