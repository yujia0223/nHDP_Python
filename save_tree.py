#!/usr/bin/env python3


import csv
from heapq import nlargest
import pandas as pd


def load_vocab(vocab_path):
    with open(vocab_path, encoding='utf-8') as f:
        return [line.strip() for line in f]


def load_tree(tree_csv_path):
    csv.field_size_limit(int(2**31 - 1))
    with open(tree_csv_path) as f:
        nodes = dict(
            (
                tuple(int(x) for x in row['me'].strip().split()),
                dict(parent_loc=tuple(int(x) for x in row['parent'].strip().split()),
                     tau_sums=float(row['tau_sums'].strip()),
                     lambda_sums=[float(x) for x in row['lambda_sums'].strip().split()],
                     children={})
            )
            for row in csv.DictReader(f)
        )

    root_loc = (1,)
    if root_loc not in nodes:
        nodes[root_loc] = dict(parent_loc=(), children={})

    for (node_loc, node) in nodes.items():
        parent_loc = node['parent_loc']
        if parent_loc:
            child_idx = node_loc[-1]
            nodes[parent_loc]['children'][child_idx] = node

    for (node_loc, node) in nodes.items():
        node['me'] = node_loc
        node['children'] = [child for (_, child) in sorted(node['children'].items())]

    return nodes[root_loc]


def save_tree(tree_csv_path, vocab_path, output_path, num_words=5):
    tree = load_tree(tree_csv_path)
    vocab = load_vocab(vocab_path)
    
    # Prepare to collect tree data
    paths = []
    def traverse(node, path=[], depth=0):
        if depth not in path:
            path.append([])  # Extend the path list to hold this level's words
        top_words = [
            word
            for (_, word)
            in nlargest(
                num_words,
                [
                    (weight, word)
                    for (weight, word)
                    in zip(node.get('lambda_sums', []), vocab)
                ]
            )
        ]
        path[depth].append(' '.join(top_words))
        
        # If this node has children, continue traversing
        if node['children']:
            for child in node['children']:
                traverse(child, list(path), depth + 1)  # Pass a copy of the path
        else:
            # Leaf node: finalize this path
            while len(path) < len(paths[0]) if paths else 0:
                path.append([""] * len(path[0]))  # Fill empty levels for alignment
            paths.append(path)
    
    # Initialize traversal
    traverse(tree)
    
    # Flatten paths to fit the DataFrame format
    data = []
    for path in paths:
        row = []
        for level_words in path:
            # print(level_words[0].split(' '))
            row.extend(level_words)
        data.append(row)
    
    # The maximum depth determines the number of columns
    columns = [f'level{i}' for i in range(len(data[0]))]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    
    return df

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Load tree from CSV file and print topics.')
    parser.add_argument('tree_csv_path')
    parser.add_argument('vocab_path')
    parser.add_argument('output_path')
    parser.add_argument('--num-words', '-n', type=int, default=5)
    # parser.add_argument('--interactive', '-i', action='store_true')
    args = parser.parse_args()
    save_tree(args.tree_csv_path, args.vocab_path, args.output_path,
               num_words=args.num_words)




if __name__ == '__main__':
    main()
