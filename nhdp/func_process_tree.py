import numpy as np
from scipy.special import psi

def rev_cumsum(a):
    a = np.flipud(a)
    vec = np.cumsum(a)
    vec = np.flipud(vec)

    return vec
    
        
def func_process_tree(Tree=None, beta0=None, gamma1=None):
    # process the tree for the current batch
    # (put info from Tree struct into matrix and vector form)

    godel = np.log([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    godel = godel.reshape(1, -1)
    Voc = len(Tree['lambda_sums'][0])  # lambda sum
    total_num_topics = len(Tree['lambda_sums'])  # total num of topics

    id_parent = np.zeros(total_num_topics)  # real ids of parents (K*1)
    id_me = np.zeros(total_num_topics)  # real ids of topics (K*1)
    ElnB = np.zeros((total_num_topics, Voc))  # real ids of parents (K*1)
    count = np.zeros(total_num_topics)  # real ids of parents (K*1)


    for i in range(len(Tree['lambda_sums'])):
        # change the real id to numeric value
        # unique real id of parent of node i
        id_parent[i] = np.dot((Tree['parent'][i]+1).reshape(1,-1), np.transpose(godel[:,0:len(Tree['parent'][i])])) #TODO:confirm if need to +1 differ at i=0
        # unique real id of node i
        id_me[i] = np.dot((Tree['me'][i]+1).reshape(1,-1), np.transpose(godel[:,0:len(Tree['me'][i])]))
        # fill in row of Elogtheta
        ElnB[i, :] = psi(Tree['lambda_sums'][i] + beta0) - psi(sum(Tree['lambda_sums'][i] + beta0))
        # fill in element of doc assignment counts
        count[i] = Tree['tau_sums'][i]

    ElnPtop = np.zeros(total_num_topics)  # Elogp (global Elogpi) (k * 1) # TODO: confirm
    groups = np.unique(id_parent)  # set of real parent ids

    for g in range(len(groups)):
        # find integer indices of this node's children
        group_idx = np.argwhere(id_parent == groups[g]).reshape(-1)
        this = count[group_idx]  # get doc assignment counts of children (35,1)

        group_count = sorted(this, reverse=True)  # sort children by doc assignment counts
        sort_group_idx = np.flip(np.argsort(this))  # decending order of index

        a = np.array(group_count) + 1  # estimate tau1 (1 * num children)
        b = np.append(rev_cumsum(group_count[1:]),0) + gamma1
        ElnV = psi(a) - psi(a + b)  # compute ElogV
        Eln1_V = psi(b) - psi(a + b)  # compute Elog(1-V)
        vec = ElnV + np.insert(np.cumsum(Eln1_V[0:-1]), 0, [0], axis=0)
        ElnPtop[group_idx[sort_group_idx]] = vec

    return ElnB, ElnPtop, id_parent, id_me