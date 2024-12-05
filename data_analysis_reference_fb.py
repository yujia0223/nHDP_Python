import pandas as pd
import matplotlib.pyplot as plt

# Load the first few rows of the file
file_path = "E:/course-phd/202009-project1/hlda/benchmark/KGESemanticAnalysis-main/data/freebase/freebaseTypes.tsv"

# data_sample = pd.read_csv(file_path, delimiter='\t', nrows=5)
# Load the entire file
data = pd.read_csv(file_path, delimiter='\t', header=None, names=['Identifier', 'Descriptor'])

# Analysis
unique_identifiers = data['Identifier'].nunique() # 12615
unique_descriptors = data['Descriptor'].nunique() #2260
print('unique_identifiers: ', unique_identifiers)
print('unique_descriptors: ', unique_descriptors)

# Most frequent descriptors
most_frequent_descriptors = data['Descriptor'].value_counts()
print('most_frequent_descriptors: ', most_frequent_descriptors)

# Calculate the distribution of all descriptors
descriptor_distribution_all = data['Descriptor'].value_counts()

# Plotting the distribution of all descriptors
plt.figure(figsize=(16, 8))
descriptor_distribution_all.plot(kind='bar')
plt.title("Distribution of All Descriptors/Types")
plt.xlabel("Descriptors/Types")
plt.ylabel("Frequency")
plt.xticks([])  # Hide x-tick labels for clarity
plt.grid(axis='y')
plt.show()

# Distribution of number of descriptors/types per identifier
descriptor_distribution = data.groupby('Identifier').size().value_counts().sort_index()
print('descriptor_distribution: ', descriptor_distribution)





# # Plotting the histogram for number of descriptors/types per identifier
# plt.figure(figsize=(14, 7))
# descriptor_distribution.plot(kind='bar')
# plt.title("Distribution of Number of Descriptors/Types per Identifier")
# plt.xlabel("Number of Descriptors/Types")
# plt.ylabel("Number of Identifiers")
# plt.grid(axis='y')
# plt.show()

# Sample identifiers with multiple descriptors/types
sample_multi_label = data[data['Identifier'].duplicated(keep=False)].groupby('Identifier').agg(list)
print('sample_multi_label: ', sample_multi_label)


# Extract identifiers with multiple descriptors
multi_label_identifiers = data[data['Identifier'].duplicated(keep=False)].groupby('Identifier').size()
# multi_label_identifiers.to_csv('multi_label_identifiers.csv')

# # Plotting the distribution of multi-label identifiers
# plt.figure(figsize=(16, 8))
# multi_label_identifiers.plot(kind='bar')
# plt.title("Identifiers with Multiple Descriptors/Types")
# plt.xlabel("Identifiers")
# plt.ylabel("Number of Descriptors/Types")
# plt.xticks([])  # Hide x-tick labels for clarity
# plt.grid(axis='y')
# plt.show()

print('multi_label_identifiers: ', multi_label_identifiers)

# Filter identifiers with fewer than 10 descriptors
identifiers_less_than_10 = multi_label_identifiers[multi_label_identifiers < 10]

# # Plotting the distribution of these filtered identifiers
# plt.figure(figsize=(16, 8))
# identifiers_less_than_10.plot(kind='bar')
# plt.title("Identifiers with Fewer than 10 Descriptors/Types")
# plt.xlabel("Identifiers")
# plt.ylabel("Number of Descriptors/Types")
# plt.xticks([])  # Hide x-tick labels for clarity
# plt.grid(axis='y')
# plt.show()

# identifiers_less_than_10

# Extracting descriptors for the filtered identifiers
descriptors_for_filtered_ids = data[data['Identifier'].isin(identifiers_less_than_10.index)].groupby('Identifier').agg(list)

# Displaying the descriptors for these identifiers
print(descriptors_for_filtered_ids)
descriptors_for_filtered_ids.to_csv('descriptors_for_filtered_ids.csv')

# Save the descriptor distribution to a CSV file
output_path = "descriptor_distribution.csv"