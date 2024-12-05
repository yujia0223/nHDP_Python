import math
from collections import Counter
import pandas as pd
import numpy as np
from gensim import corpora

EPSILON = 1e-12

def calculate_coverage(topics, texts):
    # # Flatten the list of documents into a single list of words
    # all_words = [word for doc in texts for word in doc]
    # # print('all words in original texts',all_words)
    # word_counts = Counter(all_words)

    # Replace all top words in the documents with the first word
    topics_W1 = [] # first word of different topics in level L
    coverage_l = []
    
    for topic in topics:
        #   print(topic)
        # print(topic)
        # print(words)
        if topic:
            first_word = topic[0]
            replaced_docs = [[first_word if word in topic else word for word in doc] for doc in texts] # w_1^z
            topics_W1.append(first_word)

            # Flatten the list of replaced documents into a single list of words
            # print('replaced docs',replaced_docs)
            all_replaced_words = [word for doc in replaced_docs for word in doc]
            # print('replaced words',all_replaced_words)
            replaced_word_counts = Counter(all_replaced_words)
            joint_prob_w1_wj = np.zeros((len(topics), len(replaced_word_counts)))
            for doc in replaced_docs:
                for word in doc:
                    if first_word in doc and word != first_word:
                        i = topics.index(topic)
                        j = list(replaced_word_counts.keys()).index(word)
                        joint_prob_w1_wj[i, j] += 1

            # Calculate PMI for each word in the topic

            # for first_word in topics_W1:
            pmi_scores = []
            for word in all_replaced_words:
                # print(word)
                p_w1 = replaced_word_counts[topic[0]] / len(all_replaced_words)
                p_wj = replaced_word_counts[word] / len(all_replaced_words)
                p_w1_wj = joint_prob_w1_wj[topics.index(topic), list(replaced_word_counts.keys()).index(word)] / len(all_replaced_words)

                # print(p_w1,p_wj,p_w1_wj)
                pmi = math.log(p_w1_wj / (p_w1 * p_wj + EPSILON) + EPSILON) # math error
                pmi_scores.append(pmi)

            # Calculate coverage for the topic
            coverage = sum(pmi_scores) / len(pmi_scores)
            coverage_l.append(coverage)
    # print(coverage_l)
    coverage_score_level = sum(coverage_l)/len(coverage_l)

    return coverage_score_level

def calculate_coverage_vocab(topics, texts, level): # word level same as calculate coverage
    # Flatten the list of documents into a single list of words
    all_words = [word for doc in texts for word in doc]
    # print('all words in original texts',all_words)
    word_counts = Counter(all_words)

    # Replace all top words in the documents with the first word
    topics_W1 = [] # first word of different topics in level L
    topics_coverage = []
    marginal_prob_w1 = np.zeros(len(topics))
    for topic in topics:
        #   print(topic)
        # print(topic)
        # print(words)
        if topic:
            first_word = topic[0]
            replaced_docs = [[first_word if word in topic else word for word in doc] for doc in texts] # w_1^z
            topics_W1.append(first_word)

            # Flatten the list of replaced documents into a single list of words
            # print('replaced docs',replaced_docs)
            all_replaced_words = [word for doc in replaced_docs for word in doc]
            # print('replaced words',all_replaced_words)
            replaced_word_counts = Counter(all_replaced_words)

            joint_prob_w1_wj = np.zeros((len(topics), len(replaced_word_counts))) # need to be done here initialize for each topic
            marginal_prob_wj = np.zeros(len(replaced_word_counts))

            for doc in replaced_docs:
                for word in doc:
                    i = topics.index(topic)
                    j = list(replaced_word_counts.keys()).index(word)
                    marginal_prob_wj[j] += 1
                    if word != first_word:
                        joint_prob_w1_wj[i, j] += 1
                    elif word == first_word:
                        marginal_prob_w1[i] += 1

            # Calculate PMI for each word in the topic

            # for first_word in topics_W1:
            pmi_scores = []
            for word in all_replaced_words:
                # print(word)
                p_w1 = marginal_prob_w1[topics.index(topic)] / len(all_replaced_words)
                p_wj = marginal_prob_wj[list(replaced_word_counts.keys()).index(word)] / len(all_replaced_words)
                p_w1_wj = joint_prob_w1_wj[topics.index(topic), list(replaced_word_counts.keys()).index(word)] / len(all_replaced_words)

                # print(p_w1,p_wj,p_w1_wj)
                pmi = math.log(p_w1_wj / (p_w1 * p_wj + EPSILON) + EPSILON) # math error
                pmi_scores.append(pmi)

            # Calculate coverage for the topic
            coverage = sum(pmi_scores) / len(pmi_scores)
            topics_coverage.append(coverage)

    if topics_coverage:
        min_coverage = min(topics_coverage)
        max_coverage = max(topics_coverage)
        avg_coverage = sum(topics_coverage) / len(topics_coverage)
    else:
        min_coverage, max_coverage, avg_coverage = 0, 0, 0  # Handle case with no topics or no coverage

    return {
        "level": level,
        "min_coverage": min_coverage,
        "max_coverage": max_coverage,
        "average_coverage": avg_coverage
    }

def calculate_coverage_doc(topics, texts, level): # doc level,
    # Flatten the list of documents into a single list of words
    all_words = [word for doc in texts for word in doc]
    # print('all words in original texts',all_words)
    word_counts = Counter(all_words)

    # Replace all top words in the documents with the first word
    topics_W1 = [] # first word of different topics in level L
    topics_coverage = []
    joint_prob_w1_wj = np.zeros((len(topics), len(word_counts)))
    marginal_prob_w1 = np.zeros(len(topics))
    marginal_prob_wj = np.zeros(len(word_counts))
    for topic in topics:
        #   print(topic)
        # print(topic)
        # print(words)
        if topic:
            first_word = topic[0]
            replaced_docs = [[first_word if word in topic else word for word in doc] for doc in texts] # w_1^z
            topics_W1.append(first_word)

            # Flatten the list of replaced documents into a single list of words
            # print('replaced docs',replaced_docs)
            all_replaced_words = [word for doc in replaced_docs for word in doc]
            # print('replaced words',all_replaced_words)
            replaced_word_counts = Counter(all_replaced_words)

            pmi_scores = []
            num_docs = len(replaced_docs)
            for word in replaced_word_counts:

                i = topics.index(topic)
                j = list(replaced_word_counts.keys()).index(word)

                for doc in replaced_docs:

                    if first_word in doc:
                        marginal_prob_w1[i] += 1

                    if word in doc:
                        marginal_prob_wj[j] += 1

                    if first_word in doc and word in doc:
                        joint_prob_w1_wj[i, j] += 1

                # Calculate PMI for each word in the documents
                
                # pmi = math.log((joint_prob_w1_wj[i,j]*num_docs)/ (marginal_prob_w1[i]*marginal_prob_wj[j] + EPSILON)+ EPSILON)
                         
                numerator = (joint_prob_w1_wj[i,j] / num_docs) + EPSILON
                denominator = (marginal_prob_w1[i] / num_docs) * (marginal_prob_wj[j] / num_docs + EPSILON)
                pmi = np.log(numerator / denominator + EPSILON)
                pmi_scores.append(pmi)
            # print('pmi scores',pmi_scores)
            # print(joint_prob_w1_wj)
            # print(marginal_prob_w1)
            # print(marginal_prob_wj)
            coverage = sum(pmi_scores) / len(pmi_scores)
            topics_coverage.append(coverage)

    if topics_coverage:
        min_coverage = min(topics_coverage)
        max_coverage = max(topics_coverage)
        avg_coverage = sum(topics_coverage) / len(topics_coverage)
    else:
        min_coverage, max_coverage, avg_coverage = 0, 0, 0  # Handle case with no topics or no coverage

    return {
        "level": level,
        "min_coverage": min_coverage,
        "max_coverage": max_coverage,
        "average_coverage": avg_coverage
    }


def calculate_coverage_simple(topics, texts, level):
    topics_coverage = []
    for topic in topics:
        count = 0
        if topic:
            first_word = topic[0]
            replaced_docs = [[first_word if word in topic else word for word in doc] for doc in texts]
            for doc in replaced_docs:
                word_counts = Counter(doc)
                if word_counts[first_word] > 5:
                    count += 1

            # Store the coverage percentage for each topic
            topics_coverage.append(count / len(replaced_docs))

    if topics_coverage:
        min_coverage = min(topics_coverage)
        max_coverage = max(topics_coverage)
        avg_coverage = sum(topics_coverage) / len(topics_coverage)
    else:
        min_coverage, max_coverage, avg_coverage = 0, 0, 0  # Handle case with no topics or no coverage

    return {
        "level": level,
        "min_coverage": min_coverage,
        "max_coverage": max_coverage,
        "average_coverage": avg_coverage
    }

# # # tree 
# tree_1 = {
#     'root': [['game', 'play', 'team', 'going', 'think'],['game', 'play', 'team', 'going', 'think'], ['game', 'play', 'team', 'going', 'think']],
#     'level1': [['baseketball', 'season', 'game','team', 'knicks'], ['baseketball', 'season', 'game','team', 'knicks'], ['season', 'basketball','game', 'league','series']],
#     'level2': [['knicks','ewing','riley','game','patrick'],['points', 'nets', 'game','scored', 'half'], ['run', 'inning', 'hit', 'game', 'runs']],
# }

# tree_2 = {
#     'root': [['game', 'play', 'team', 'going', 'think'],['game', 'play', 'team', 'going', 'think'], ['game', 'play', 'team', 'going', 'think']],
#     'level1': [['baseketball', 'season', 'game','team', 'knicks'], ['baseketball', 'season', 'game','team', 'knicks'], ['season', 'basketball','game', 'league','series']],
#     'level2': [['knicks','ewing','riley','game','patrick'],['points', 'nets', 'game','scored', 'half'], []],
# }

# tree_3 = {
#     'root': [['game', 'play', 'team', 'going', 'think'],['game', 'play', 'team', 'going', 'think'], ['game', 'play', 'team', 'going', 'think']],
#     'level1': [['baseketball', 'season', 'game','team', 'knicks'], ['baseketball', 'season', 'game','team', 'knicks'], ['season', 'basketball','game', 'league','series']],
#     'level2': [['knicks','ewing','riley','game','patrick'],['points', 'nets', 'game','scored', 'half'], ['penalty','fault','kick','out','stadium']],
# }

# good_tree = {
#     'root': [['game', 'play', 'team', 'going', 'think'],['game', 'play', 'team', 'going', 'think'], ['game', 'play', 'team', 'going', 'think']],
#     'level1': [['baseketball', 'season', 'game','team', 'knicks'], ['baseketball', 'season', 'game','team', 'knicks'], ['season', 'basketball','game', 'league','series']],
#     'level2': [['knicks','ewing','riley','game','patrick'],['points', 'nets', 'game','scored', 'half'], ['run', 'inning', 'hit', 'game', 'runs']],
# }

# bad_tree = {
#     'root': [['game', 'team', 'play', 'season', 'knicks'], ['game', 'team', 'play', 'season', 'knicks'], ['game', 'team', 'play', 'season', 'knicks']],
#     'level1': [['basketball', 'season', 'game', 'team', 'knicks'], ['basketball', 'season', 'game', 'team', 'knicks'], ['season', 'basketball', 'game', 'team', 'knicks']],
#     'level2': [['knicks', 'ewing', 'riley', 'game', 'points'], ['points', 'nets', 'game', 'scored', 'half'], ['game', 'team', 'play', 'season', 'knicks']],
# }


# documents = [
#     "The game was intense, both teams going back and forth. The play was aggressive, each team thinking two steps ahead. The basketball season was in full swing, with each game carrying significant weight. The Knicks, in particular, were having a standout season. Patrick Ewing, under the guidance of coach Riley, was a force on the court. The game against the Nets was a highlight, with the Knicks scoring points in both halves.",
#     "As the basketball season progressed, the Knicks emerged as a team to watch. The games were thrilling, with the team's play strategy keeping fans on the edge of their seats. The Knicks' game against the Nets was a turning point in the season. Ewing's performance was exceptional, and coach Riley's game plan was executed flawlessly. The points scored in the first half set the tone for the rest of the game.",
#     "The basketball season was heating up, and the Knicks were at the center of it all. Each game was a testament to the team's hard work and strategic play. The team was going strong, with the players thinking ahead and making smart moves on the court. The game against the Nets was a memorable one, with Ewing leading the charge under Riley's guidance. The points scored in both halves were a testament to the Knicks' dominance.",
#     "The Knicks were the talk of the basketball season. Their game strategy was unmatched, and the team's play was a spectacle to watch. The team was going from strength to strength, with each game bringing them closer to their goal. The game against the Nets was a standout, with Ewing and Riley leading the team to victory. The points scored in the second half sealed the win for the Knicks.",
#     "The basketball season was in full swing, and the Knicks were making their mark. The team's play was strategic, with each game bringing new challenges and opportunities. The Knicks were going strong, with the players thinking on their feet and making smart moves. The game against the Nets was a highlight of the season, with Ewing's performance and Riley's coaching leading the team to victory. The points scored in the first half set the tone for the rest of the game.",
#     "In the heart of the season, every game unfolds like a storied chapter, rich with the crack of the bat, the strategic dash between bases, and the roar of the crowd that fills the air. As the innings progress, each hit and run scored tells a tale of determination and skill, a ballet of precision and power played out on the diamond. The essence of baseball is captured not just in the statistics of runs batted in or the elegance of a perfect pitch, but in the moments of suspense and triumph that define each game. It's a testament to the enduring allure of America's favorite pastime, where every player contributes to the narrative, weaving a legacy of athletic prowess and strategic genius."
#     "As the basketball season advances, the intensity on the court reaches fever pitch. Each game becomes a battleground where strategy, skill, and teamwork collide. The players, thinking two steps ahead, navigate the court with a blend of athleticism and tactical play, their movements a testament to hours of practice and dedication. The team's cohesion is palpable, each member playing a crucial role in both offense and defense. In this league, the series of games are not just contests of physical ability, but of mental agility and strategic planning. With every play, the team showcases their collective resolve, pushing forward, thinking, and rethinking every move to outsmart their opponents. It's a display of the beautiful game of basketball, where every dribble, pass, and shot is a step towards victory, driven by a shared goal and an unbreakable team spirit."
#     ]

# # good_tree = {
# #     'root':[['machine learning', 'algorithms', 'data', 'prediction', 'training'],['machine learning', 'algorithms', 'data', 'prediction', 'training'], ['machine learning', 'algorithms', 'data', 'prediction', 'training']],
# #     'level1': [['classification', 'regression', 'support vector machines', 'neural networks', 'decision trees'],['clustering', 'dimensionality reduction', 'k-means', 'PCA', 'autoencoders'],['agents', 'environments', 'policy', 'reward', 'Q-learning']],
# #     'level2': [['image recognition', 'spam detection', 'predictive modeling', 'speech recognition', 'stock prediction'], ['customer segmentation', 'feature extraction', 'anomaly detection', 'market basket analysis', 'recommendation systems'], ['game AI', 'robot navigation', 'portfolio management', 'dynamic pricing', 'personalized learning']],
# #     }

# # tree_df = pd.DataFrame(good_tree)
# # tree_df = pd.DataFrame(bad_tree)
# tree_df = pd.DataFrame(tree_1)
# # tree_df = pd.DataFrame(tree_2)
# # tree_df = pd.DataFrame(tree_3)
# print('hierarchy tree is \n',tree_df)

# texts = [
#     [word for word in document.lower().split()]
#     for document in documents
# ]

# dictionary = corpora.Dictionary(texts) # id2word
# corpus = [dictionary.doc2bow(text) for text in texts]
# # print(corpus)
# # print(dictionary)

# coverage_level_simple = []
# coverage_level = []
# coverage_level_doc = []
# coverage_level_vocab = []

# for level in tree_df.columns:
#     # # Convert each column to a list of tuples
#     # topics = [tuple(x) for x in tree_df[level].tolist()]


#     # # Get unique topics by converting the list to a set, then convert back to a list
#     # unique_topics = set(topics)

#     # # Print the unique topics
#     # print(list(unique_topics))
#     print()
#     unique_topics = tree_df[level].drop_duplicates().tolist()
#     print(f'topics in level {level}',unique_topics)

#     # Compute coherence and topic diversity for each list of unique topics
#     coverage_score_simple = calculate_coverage_simple(unique_topics, texts, level)
#     coverage_score = calculate_coverage(unique_topics, texts)
#     coverage_score_doc = calculate_coverage_doc(unique_topics, texts, level)
#     coverage_score_vocab = calculate_coverage_vocab(unique_topics, texts, level)

#     coverage_level.append(coverage_score)
#     coverage_level_simple.append(coverage_score_simple)
#     coverage_level_doc.append(coverage_score_doc)
#     coverage_level_vocab.append(coverage_score_vocab)

# print('coverage for the hierarchical levels vocab count',coverage_level)
# print('coverage for the hierarchical levels simple docs count',coverage_level_simple)
# print('coverage for the hierarchical levels docs count',coverage_level_doc)
# print('coverage for the hierarchical levels vocab count 2',coverage_level_vocab)