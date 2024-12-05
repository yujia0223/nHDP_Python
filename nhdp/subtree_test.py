import pandas as pd

new_data_df = pd.read_csv('data/fb15k-237/all_triples_extracted_po.txt', index=False, header = False)


new_data_df.columns = ['s','p','o']
print('the 1st document is: \n')
print(new_data_df[new_data_df.s == all_subjects[0]])
print(list(new_data_df[new_data_df.s == all_subjects[0]].p))
print('the 2nd document is: \n')
print(new_data_df[new_data_df.s == all_subjects[1]])
print(list(new_data_df[new_data_df.s == all_subjects[1]].p))
print('the 99th document is: \n')
print(new_data_df[new_data_df.s == all_subjects[99]])
print(list(new_data_df[new_data_df.s == all_subjects[99]].p))

new_data_df.columns = ['s','p','o']
print('the 1st document is: \n')
print(new_data_df[new_data_df.s == all_subjects[0]])
print(list(new_data_df[new_data_df.s == all_subjects[510]].p))
print('the 2nd document is: \n')
print(new_data_df[new_data_df.s == all_subjects[1]])
print(list(new_data_df[new_data_df.s == all_subjects[1050]].p))
print('the 99th document is: \n')
print(new_data_df[new_data_df.s == all_subjects[99]])
print(list(new_data_df[new_data_df.s == all_subjects[4335]].p))