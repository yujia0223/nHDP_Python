def calculate_interest_based_coherence(utility_scores, frequency_scores, alpha=0.5):
    # Initialize an empty dictionary to store the coherence scores
    coherence_scores = {}

    # Iterate over each node in the utility_scores and frequency_scores
    for node in utility_scores.keys():
        # Calculate the coherence score for the node
        coherence_score = alpha * utility_scores[node] + (1 - alpha) * frequency_scores[node]

        # Add the coherence score to the dictionary
        coherence_scores[node] = coherence_score

    return coherence_scores


############################################# results prepration #############################################
# tree 
tree = {
    'root': [['game', 'play', 'team', 'going', 'think'],['game', 'play', 'team', 'going', 'think'], ['game', 'play', 'team', 'going', 'think']],
    'level1': [['baseketball', 'season', 'game','team', 'knicks'], ['baseketball', 'season', 'game','team', 'knicks'], ['season', 'basketball','game', 'league','series']],
    'level2': [['knicks','ewing','riley','game','patrick'],['points', 'nets', 'game','scored', 'half'], ['run', 'inning', 'hit', 'game', 'runs']],
}


documents = [
    "The game was intense, both teams going back and forth. The play was aggressive, each team thinking two steps ahead. The basketball season was in full swing, with each game carrying significant weight. The Knicks, in particular, were having a standout season. Patrick Ewing, under the guidance of coach Riley, was a force on the court. The game against the Nets was a highlight, with the Knicks scoring points in both halves.",
    "As the basketball season progressed, the Knicks emerged as a team to watch. The games were thrilling, with the team's play strategy keeping fans on the edge of their seats. The Knicks' game against the Nets was a turning point in the season. Ewing's performance was exceptional, and coach Riley's game plan was executed flawlessly. The points scored in the first half set the tone for the rest of the game.",
    "The basketball season was heating up, and the Knicks were at the center of it all. Each game was a testament to the team's hard work and strategic play. The team was going strong, with the players thinking ahead and making smart moves on the court. The game against the Nets was a memorable one, with Ewing leading the charge under Riley's guidance. The points scored in both halves were a testament to the Knicks' dominance.",
    "The Knicks were the talk of the basketball season. Their game strategy was unmatched, and the team's play was a spectacle to watch. The team was going from strength to strength, with each game bringing them closer to their goal. The game against the Nets was a standout, with Ewing and Riley leading the team to victory. The points scored in the second half sealed the win for the Knicks.",
    "The basketball season was in full swing, and the Knicks were making their mark. The team's play was strategic, with each game bringing new challenges and opportunities. The Knicks were going strong, with the players thinking on their feet and making smart moves. The game against the Nets was a highlight of the season, with Ewing's performance and Riley's coaching leading the team to victory. The points scored in the first half set the tone for the rest of the game.",
    ]

tree_df = pd.DataFrame(tree)
print(tree_df)

texts = [
    [word for word in document.lower().split()]
    for document in documents
]

# dictionary = corpora.Dictionary(texts) # id2word
# corpus = [dictionary.doc2bow(text) for text in texts]
# print(corpus)
# print(dictionary)

############################################# evaluation #############################################
# print(tree_df)
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

print('child_relatedness',child_relatedness)
print('non_child_relatedness',non_child_relatedness)