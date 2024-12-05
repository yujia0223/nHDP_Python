import pandas as pd

# Loading the entire reference CSV file to extract unique entities
reference_df = pd.read_csv("data/fb15k-237/freebase_reference_data_nhdp.csv")
unique_entities_reference = set(reference_df['entities'])

# Reading the vocab_o.txt file to extract unique values
with open("data/fb15k-237/vocab_o.txt", "r") as vocab_file:
    vocab_data = vocab_file.readlines()

# Extracting unique values from vocab_o.txt
unique_vocab_values = set(vocab_data)

# Comparing the unique entities from vocab_o.txt with the unique entities from the reference file
matching_vocab_entities = unique_vocab_values.intersection(unique_entities_reference)

# Getting the number of matching unique entities and the last value from vocab_o.txt
number_of_matching_vocab_entities = len(matching_vocab_entities)
last_value_vocab = vocab_data[-1].strip()

print(number_of_matching_vocab_entities, last_value_vocab)

# Calculating the sum of the number of unique entities in corpus_o.txt and unique vocab in vocab_o.txt
total_unique_values = len(unique_entities_reference)
total_unique_vocab = len(unique_vocab_values)

sum_of_entities_and_vocab = total_unique_values + total_unique_vocab
print(sum_of_entities_and_vocab)

