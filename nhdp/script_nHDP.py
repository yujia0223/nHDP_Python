import numpy as np
import pandas as pd
from nHDP_init import nHDP_init
from nHDP_step import nHDP_step
from datetime import datetime



if __name__ == '__main__':

    # np.random.seed(0) # 0, 1, 1234, # ok :7, 30
    # print(np.random.get_state())
    # data input
    df = pd.read_csv("data/fb15k-237/corpus_po_20240102172952.txt", sep=" ", header=None)
    num_doc = len(np.unique(df[0].values))
    # print(num_doc)
    X_index = {}
    X_Count = {}
    for i in range(num_doc):
        # X_index[i] = df[df[0] == i][1].values.reshape(1, -1)
        # X_Count[i] = df[df[0] == i][2].values.reshape(1, -1)
        X_index[i] = df[df[0] == i][1].values
        X_Count[i] = df[df[0] == i][2].values
        # check the consistency
        # print(i)
        # print(X_index[i].shape)
        # print(X_Count[i].shape)
        if len(X_index[i]) != len(X_Count[i]):
            print('fail')



    # initialize / use a large subset of documents (e.g., 10,000) contained in Xid and Xcnt to initialize

    num_topics = np.array([2, 2])  # first level, second level children of each of first level node, third level
    scale = 100000
    batch_size = 20
    num_iters = 10 # 1000 test in 100

    Tree = nHDP_init(X_index, X_Count, num_topics, scale)
    # TODO: the second value should be 47, but here just 44/20????
    # Tree = pd.DataFrame(Tree)
    for i in range(len(Tree['lambda_sums'])):
        if Tree['tau_sums'][i] == 0:
            Tree['lambda_sums'][i] = 0
        vec = np.random.gamma(np.ones(len(Tree['lambda_sums'][0]))) #TODO: change the length the W number of word count
        Tree['lambda_sums'][i] = 0.95 * Tree['lambda_sums'][i] + 0.05 * scale * vec / sum(vec) # setting with copy warning if pandas
        # check errors
        # if Tree['lambda_sums'][i].shape[0] != 12050:
        #     # print(i)
        #     # print(Tree['lambda_sums'][i].shape)

    # main loop / to modify this, at each iteration send in a new subset of docs
    # contained in Xid_batch and Xcnt_batch
    beta0 = 0.1 # this parameter is the Dirichlet base distribution and can be played with
    for i in range(num_iters):
        # print(i)
        tempt = np.random.rand(len(X_index))
        a = np.sort(tempt)
        b = np.argsort(tempt)
        rho = (1 + i) ** - 0.75 # step size can also be played with

        Xid_batch = {a: X_index[a] for a in b[0:batch_size] if a in X_index}
        # Xid_batch = X_index[b[0:batch_size]]
        Xcnt_batch = {a: X_Count[a] for a in b[0:batch_size] if a in X_Count}
        # Xcnt_batch = X_Count[b[0:batch_size]]
        Tree = nHDP_step(Xid_batch,Xcnt_batch,Tree,scale,rho,beta0)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    Tree_df = pd.DataFrame(Tree)
    print(Tree_df)
    Tree_df.to_csv(f'{timestamp}_nhdp_fb_python_{num_topics}_{beta0}.csv', index = False)