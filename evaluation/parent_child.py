import math
import pandas as pd
# import numpy as np
from collections import Counter
EPSILON = 1e-12

def calculate_relatedness(parent_topic, child_topics, texts):
    # child_topics could be non child topics too from input
  
    # Replace all child words in the texts with the topic word
    replaced_docs_p = [[parent_topic[0] if word in parent_topic else word for word in doc] for doc in texts]

    # Flatten the list of replaced texts into a single list of words
    all_replaced_words_p = [word for doc in replaced_docs_p for word in doc]
    replaced_word_counts_p = Counter(all_replaced_words_p)

    # Calculate D for each child and non-child topic
    relatedness_scores = []
    for child_t in child_topics:
        co_pc = 0
          # Flatten the list of texts into a single list of words
        replaced_docs_pc = [[parent_topic[0] if word in parent_topic and child_t else word for word in doc] for doc in texts]
        all_words_pc = [word for doc in replaced_docs_pc for word in doc]
        # print('all words in original texts',all_words_pc)
        all_replaced_words_pc = Counter(all_words_pc)
        for doc in texts:
            for word in doc:
                if word in parent_topic and word in child_t:
                    co_pc += 1

        # Replace all child words in the texts with the topic word
        replaced_docs_c = [[child_t[0] if word in child_t else word for word in doc] for doc in texts]

        # Flatten the list of replaced texts into a single list of words
        all_replaced_words_c = [word for doc in replaced_docs_c for word in doc]
        replaced_word_counts_c = Counter(all_replaced_words_c)

        # Calculate D for each child and non-child topic

        D_z_chnz_k = co_pc / len(all_replaced_words_pc) # parent and child
        D_z = replaced_word_counts_p[parent_topic[0]] / len(all_replaced_words_p) # parent
        D_chnz_k = replaced_word_counts_c[child_t[0]] / len(all_replaced_words_c) # child or non child topic

        # Calculate the relatedness score
        relatedness_score = math.log(D_z_chnz_k / (D_z * D_chnz_k) + EPSILON)
        relatedness_scores.append(relatedness_score)

    relatedness_score_level = sum(relatedness_scores) / (len(relatedness_scores)+ EPSILON)

 
    return relatedness_score_level

# # ############################################# results prepration #############################################
# # tree 
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
# # # tree 
# # good_tree = {
# #     'root': [['game', 'play', 'team', 'going', 'think'],['game', 'play', 'team', 'going', 'think'], ['game', 'play', 'team', 'going', 'think']],
# #     'level1': [['baseketball', 'season', 'game','team', 'knicks'], ['baseketball', 'season', 'game','team', 'knicks'], ['season', 'basketball','game', 'league','series']],
# #     'level2': [['knicks','ewing','riley','game','patrick'],['points', 'nets', 'game','scored', 'half'], ['run', 'inning', 'hit', 'game', 'runs']],
# # }

# # bad_tree = {
# #     'root': [['game', 'team', 'play', 'season', 'knicks'], ['game', 'team', 'play', 'season', 'knicks'], ['game', 'team', 'play', 'season', 'knicks']],
# #     'level1': [['basketball', 'season', 'game', 'team', 'knicks'], ['basketball', 'season', 'game', 'team', 'knicks'], ['season', 'basketball', 'game', 'team', 'knicks']],
# #     'level2': [['knicks', 'ewing', 'riley', 'game', 'points'], ['points', 'nets', 'game', 'scored', 'half'], ['game', 'team', 'play', 'season', 'knicks']],
# # }

# documents = [
#     "The game was intense, both teams going back and forth. The play was aggressive, each team thinking two steps ahead. The basketball season was in full swing, with each game carrying significant weight. The Knicks, in particular, were having a standout season. Patrick Ewing, under the guidance of coach Riley, was a force on the court. The game against the Nets was a highlight, with the Knicks scoring points in both halves.",
#     "As the basketball season progressed, the Knicks emerged as a team to watch. The games were thrilling, with the team's play strategy keeping fans on the edge of their seats. The Knicks' game against the Nets was a turning point in the season. Ewing's performance was exceptional, and coach Riley's game plan was executed flawlessly. The points scored in the first half set the tone for the rest of the game.",
#     "The basketball season was heating up, and the Knicks were at the center of it all. Each game was a testament to the team's hard work and strategic play. The team was going strong, with the players thinking ahead and making smart moves on the court. The game against the Nets was a memorable one, with Ewing leading the charge under Riley's guidance. The points scored in both halves were a testament to the Knicks' dominance.",
#     "The Knicks were the talk of the basketball season. Their game strategy was unmatched, and the team's play was a spectacle to watch. The team was going from strength to strength, with each game bringing them closer to their goal. The game against the Nets was a standout, with Ewing and Riley leading the team to victory. The points scored in the second half sealed the win for the Knicks.",
#     "The basketball season was in full swing, and the Knicks were making their mark. The team's play was strategic, with each game bringing new challenges and opportunities. The Knicks were going strong, with the players thinking on their feet and making smart moves. The game against the Nets was a highlight of the season, with Ewing's performance and Riley's coaching leading the team to victory. The points scored in the first half set the tone for the rest of the game.",
#     "In the heart of the season, every game unfolds like a storied chapter, rich with the crack of the bat, the strategic dash between bases, and the roar of the crowd that fills the air. As the innings progress, each hit and run scored tells a tale of determination and skill, a ballet of precision and power played out on the diamond. The essence of baseball is captured not just in the statistics of runs batted in or the elegance of a perfect pitch, but in the moments of suspense and triumph that define each game. It's a testament to the enduring allure of America's favorite pastime, where every player contributes to the narrative, weaving a legacy of athletic prowess and strategic genius."
#     "As the basketball season advances, the intensity on the court reaches fever pitch. Each game becomes a battleground where strategy, skill, and teamwork collide. The players, thinking two steps ahead, navigate the court with a blend of athleticism and tactical play, their movements a testament to hours of practice and dedication. The team's cohesion is palpable, each member playing a crucial role in both offense and defense. In this league, the series of games are not just contests of physical ability, but of mental agility and strategic planning. With every play, the team showcases their collective resolve, pushing forward, thinking, and rethinking every move to outsmart their opponents. It's a display of the beautiful game of basketball, where every dribble, pass, and shot is a step towards victory, driven by a shared goal and an unbreakable team spirit."
#     ]

# # tree_df = pd.DataFrame(tree_1)
# # tree_df = pd.DataFrame(tree_2)
# tree_df = pd.DataFrame(tree_3)
# # # tree_df = pd.DataFrame(good_tree)
# # tree_df = pd.DataFrame(bad_tree)
# # print(tree_df)

# texts = [
#     [word for word in document.lower().split()]
#     for document in documents
# ]

# # dictionary = corpora.Dictionary(texts) # id2word
# # corpus = [dictionary.doc2bow(text) for text in texts]
# # print(corpus)
# # print(dictionary)

# ############################################# evaluation #############################################
# # print(tree_df)
# column_list = list(tree_df.columns)
# child_relatedness = {}
# non_child_relatedness = {}
# for level in column_list[1:]:

#     print(f"Level: {level}")
#     parent_index = column_list[column_list.index(level) - 1]
#     parent_topics = tree_df[parent_index].drop_duplicates().tolist()
#     print('parent topics', parent_topics)
#     print('length of parent topics',len(parent_topics))


#     # if len(parent_topics) <= 1: # if need?
#     #     child_topics = tree_df[level].drop_duplicates().tolist()
#     #     print('child topics',child_topics)
#     # else:

#     all_topics = tree_df[level].drop_duplicates().tolist() # all topics in the level


#     for p_topic in parent_topics:
#         print(p_topic)
#         child_topics = tree_df[tree_df[column_list[column_list.index(level) - 1]].apply(lambda x: x == p_topic)][level].drop_duplicates().tolist()
#         child_topics = [topic for topic in child_topics if topic] # remove empty lists
#         print('child topics', child_topics)
#         non_child_topics = [topic for topic in all_topics if topic not in child_topics and topic]

#         print('non child topics',non_child_topics)

#         child_relatedness_t = calculate_relatedness(p_topic, child_topics, texts)
#         child_relatedness[tuple(p_topic)] = child_relatedness_t
#         print('child relatedness',child_relatedness_t)

#         # if non_child_topics: # if non child topics exist
#         non_child_relatedness_t = calculate_relatedness(p_topic, non_child_topics, texts)
#     # non_child_relatedness_t = calculate_relatedness(p_topic, non_child_topics, texts)
#         non_child_relatedness[tuple(p_topic)] = non_child_relatedness_t
#         print('non child relatedness',non_child_relatedness_t) 
#     # unique_topics = tree_df[levels].drop_duplicates().tolist()
#     # print(f'topics in level {level}',unique_topics)

#     # # Compute coherence and topic diversity for each list of unique topics
#     # relate_score = calculate_relatedness(unique_topics, texts)
#     # parent_child_relatedness.append(relate_score)

# print('child_relatedness',child_relatedness)
# print('non_child_relatedness',non_child_relatedness)