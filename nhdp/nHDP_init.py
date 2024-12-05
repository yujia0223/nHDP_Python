import numpy as np
from sklearn.cluster import KMeans
import numpy.matlib
    
# K-Means algorithm with L1 assignment and L2 mean minimization
def K_means_L1(X=None, K=None, maxite=None):
    D = X.shape[1]
    tempt = np.random.rand(D)
    a = np.sort(tempt)
    b = np.argsort(tempt)
    if D < K:
        B = np.random.rand(X.shape[0], K)
        B = B / np.matlib.repmat(np.sum(B, axis = 0), B.shape[0], 1)
    else:
        B = X[:, b[0:K]] # chose first k documents

    c = np.zeros(D)
    for ite in range(maxite):
        for d in range(D):
            tempt = np.sum(np.abs(B - np.matlib.repmat(X[:, d].reshape(-1,1), 1, K)), axis = 0)
            a =np.min(tempt) # no useful?
            c[d] = np.argmin(tempt)
        for k in range(K):
            B[:, k] = np.mean(X[:, c == k], axis = 1)


    #cnt = histc(c, np.arange(1, K + 1))

    cnt, _ = np.histogram(c, bins=np.arange(K + 1))
    t1 = np.sort(cnt)[::-1] # descending order # no useful?
    t2 = np.argsort(cnt)[::-1]
    B = B[:, t2]
    c2 = np.zeros(len(c))
    for i in range(len(t2)):
        idx = np.argwhere(c == t2[i]).reshape(-1)
        c2[idx] = i # TODO: confirm the index  problem

    c = c2
    return B, c

def nHDP_init(Xid, Xct, num_topics, scale):
    # NHDP_INIT initializes the nHDP algorithm using a tree-structured k-means algorithm.
    #

    # depth_tree = len(model_params.num_topics) # number of levels in tree (depth)
    depth_tree = len(num_topics)
    Doc = len(Xid)  # Dt
    Voc = 0  # W

    for d in range(Doc):
        Voc = max(Voc, max(Xid[d] + 1))  #TODO: confirm index problem +1
    # get the features for each doc
    print(Voc)
    # should think the index and shape carefully
    X = np.zeros((Voc, Doc))  # W x Dt normalized data matrix (probability column vectors)
    # index start from 0

    for d in range(Doc):
        X[Xid[d], d] = np.transpose(Xct[d]) / sum(Xct[d])

    # equation 16, section 4.2.1.1
    # each natural number has a unique prime factorization prod_i[(p_i)^(n_i)], so
    # each log prime factorization sum_i[n_i log p_i] produces a different real number
    godel = np.log([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])  # use to transfer the id_parent, id_me
    godel = godel.reshape(1,-1)
    cluster_assignments = np.zeros((depth_tree + 1, Doc), dtype = 'int32')  # matrix of vector indices (column vectors) representing document assignments
    cluster_assignments[0, :] = 0 # match the python index TODO: key impact to performance
    Tree = {'lambda_sums': [], 'tau_sums': [], 'parent': [], 'me': []} # think about how to use the pandas dataframe
    init_num_iters = 3

    # loop over levels
    for l in range(depth_tree):
        print(f'beginning initialization step {l}/{depth_tree}...\n');
        K = num_topics[l]  # number of topics at this level
        # godel[:,0:l+1] * cluster_assignments[0:l+1,:] MATLAB
        vec = np.dot(godel[:,0:l+1], cluster_assignments[0:l+1,:]+1)  # TODO: compute real ids of topics at this level to documents is assigned, match with matlab
        vec = vec.reshape(-1, )
        S = np.unique(vec)  # compute real ids of topics at this level to which at least one doc is assigned
        for s in range(len(S)):  # loop over topics used at this level

            idx = np.argwhere(vec == S[s]).reshape(-1)  # compute vector id of current topic
            # print('compute vector id of current topic',idx)
            X_sub = X[:, idx]  # compute subset of documents assigned here
            print('subset', X_sub.shape)
            # [centroids,cluster_assignments_sub] = K_means_L1(X_sub,K,alg_params.init_num_iters) # matlab
            # kmeans = KMeans(n_clusters=K, max_iter=init_num_iters, random_state=0).fit(X_sub.T)  # D,W should be
            # centroids = kmeans.cluster_centers_
            # cluster_assignments_sub = kmeans.labels_
            centroids, cluster_assignments_sub = K_means_L1(X_sub, K, init_num_iters)

            cluster_assignments[l + 1, idx] = cluster_assignments_sub  # update assignments table, assigning current documents to children of s
            # tau_sums = histc(cluster_assignments_sub,1:K) #matlab number of topics assigned to each cluster
            tau_sums, _ = np.histogram(cluster_assignments_sub, bins=np.arange(K+1))  # number of topics assigned to each cluster

            # initialize tree nodes for children of s
            # equation 22-25
            # global
            for i in range(centroids.shape[1]):  # get the columns of the cluster
                # replace end with -1 in python
                #
                Tree['lambda_sums'].append(scale * np.transpose(centroids[:,i]))  # 1 x W theta ss # TODO why end + 1? # a little bit different, confirm if need to be reshape (1,N)
                Tree['tau_sums'].append(scale * tau_sums[i] / Doc)  # count of docs in subtree rooted here
                Tree['parent'].append(np.transpose(cluster_assignments[0:l+1, idx[0]])) # vector id of parent
                Tree['me'].append(np.append(Tree['parent'][-1],i))  # vector id

            # compute "probability of what remains"
            # subtract off mean
            for i in range(len(cluster_assignments_sub)):
                X[:, idx[i]] = X[:, idx[i]] - centroids[:,int(cluster_assignments_sub[i])]  # subtract off mean
                X[X[:, idx[i]] < 0, idx[i]] = 0  # threshold out negative values
                X[:, idx[i]] = X[:, idx[i]] / sum(X[:, idx[i]])  # renormalize

            print('Finished ', l, '/', depth_tree, ' : ', s, '/', len(S))

    # # new from others
    # print('postprocessing initialized tree...')
    # for i in range(len(Tree)):
    #     if Tree[i].tau_sums == 0:
    #         Tree[i].lambda_sums[:] = 0
    #     # generate Dirichlet rv with concentration param init_rand_scale / Voc
    #     randomness = np.stats.gamma.rvs(np.ones(1,len(Tree[i].lambda_sums)) * alg_params.init_rand_scale / Voc,scale = 1)
    #     randomness = randomness / sum(randomness)
    #     Tree[i].lambda_sums = alg_params.init_scale*(alg_params.kappa*Tree[i].lambda_sums + (1 - alg_params.kappa)*(1/Voc + randomness))
    #     Tree[i].tau_sums = (alg_params.init_scale/actual_init_size) * Tree[i].tau_sums

    return Tree
