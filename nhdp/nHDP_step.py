import numpy as np
from scipy.special import psi
import numpy.matlib
import matplotlib.pyplot as plt

from func_process_tree import func_process_tree
from func_doc_weight_up import func_doc_weight_up
    
def nHDP_step(Xid=None, Xcnt=None, Tree=None, scale=None, rho=None, beta0=None):
    # NHDP_STEP performs one step of stochastic variational inference
    # for the nested hierarchical Dirichlet process.

    # *** INPUT (mini-batch) ***
    # Xid{d} : contains vector of word indexes for document d
    # Xcnt{d} : contains vector of word counts corresponding to Xid{d}
    # Tree : current top-level of nHDP
    # rho : step size

    # Written by John Paisley, jpaisley@berkeley.edu

    Voc = len(Tree['lambda_sums'][0])  # vocabulary size w
    tot_tops = len(Tree['lambda_sums'])  # K # TODO: real meaning
    D = len(Xid)  # Dt
    size_subtree = np.zeros(D)  # kt per doc # TODO: keep this temporately
    hist_levels = np.zeros(10)  # counts of ...branch lengths?

    # batch suff stats
    # collects statistics for updating the tree
    B_up = np.zeros((tot_tops, Voc))  # suff stats for theta (K x W)
    weight_up = np.zeros(tot_tops)  #TODO: confirm if need to change suff stats for V (K x 1)

    gamma1 = 5  # top-level DP concentration
    gamma2 = 1  # second-level DP concentration
    gamma3 = 2 * (1 / 3)  # beta stopping switches
    gamma4 = 2 * (2 / 3)
    eps = 2.2204e-16

    # put info from Tree struct into matrix and vector form
    # ElnB: Elogtheta(K x W)
    # ElnPtop: Elogp with implicit reordering of nodes (1 x K)
    # id_parent: floating-point ids of node parents
    # id_me: floating-point ids of node
    ElnB, ElnPtop, id_parent, id_me = func_process_tree(Tree, beta0, gamma1)

    # Family tree indicator matrix. Tree_mat(j,i) = 1 indicates that node j is
    # along the path from node i to the root node(not including node i or root)
    Tree_mat = np.zeros((tot_tops,tot_tops))
    for i in range(tot_tops):
        bool = 1
        idx = i
        # iteratively climb up tree until we hit root
        while bool:
            idx = np.argwhere(id_me == id_parent[idx]).reshape(-1)  # get integer id of parent of idx
            if not len(idx) == 0:
                Tree_mat[idx, i] = 1
            else:
                # hit the root, stop iteration
                bool = 0

    # ELOB terms for U Elogpi prior for just level penalty # (410,)
    level_penalty = psi(gamma3) - psi(gamma3 + gamma4) + np.transpose(np.sum(Tree_mat, axis=0)) * (psi(gamma4) - psi(gamma3 + gamma4))
    # main loop

    Xid_keys = list(Xid.keys())
    for d in range(D):
        ElnB_d = ElnB[:, Xid[Xid_keys[d]]]  # doc-wise Elogtheta (K x Wd) # pick out words in document for penalty
        ElnV = psi(1) - psi(1 + gamma2)  # local ElogV prior
        Eln1_V = psi(gamma2) - psi(1 + gamma2)  # local Elog(1-V) prior
        ElnP_d = np.zeros(tot_tops) - np.inf  # Elogpi prior (initially all nodes inactive...) (K x 1) # -inf removes non-activated topics by giving them zero probability
        ElnP_d[id_parent == np.log(2)] = ElnV + psi(gamma3) - psi(gamma3 + gamma4)  # activate children of root

        # select subtree
        bool = 1
        idx_pick = []  # global indices of selected nodes
        Lbound = []  # scores (topic assignment and word observation ELBO components) of the subtrees corresponding to successively selected nodes
        vec_DPweights = np.zeros(tot_tops)  # ElnP_d minus the level penalty---that is, Elogpi prior for just the current level's local V terms (K x 1)
        # iter = 0
        while bool:
            # iter = iter +1
            # print(iter)
            idx_active = np.argwhere(ElnP_d > - np.inf).reshape(-1)  # indices of active (selected and potential) nodes
            penalty = ElnB_d[idx_active, :] + np.matlib.repmat(ElnP_d[idx_active].reshape(-1,1), 1, len(Xid[Xid_keys[d]]))  # Elogtheta + Elogpi for active nodes (Ka x Wd)
            C_act = penalty  # nu... see update below (Ka x Wd)
            penalty = np.multiply(penalty, np.matlib.repmat(Xcnt[Xid_keys[d]], penalty.shape[0],1))  # Elogtheta + Elogpi, scaled by word counts
            ElnPtop_act = ElnPtop[idx_active]  # Elogp (global pi) of active nodes
            if len(idx_pick) == 0:  # isempty in matlab
                # selecting first node, all active nodes are candidates
                score = np.sum(penalty, axis = 1) + ElnPtop_act  # Elogtheta + Elogpi + Elogp for active nodes (Ka x 1)
                temp = np.nanmax(score)
                #idx_this = np.argmax(score)
                idx_this = np.argwhere(score == temp).reshape(-1)[0]
                #temp, idx_this = np.amax(score)  # find best candidate (highest score) # TODO: check if it is same with matlab
                idx_pick.append(idx_active[idx_this])  # store global index of best candidate
                Lbound.append(temp - ElnPtop_act[idx_this])  # append likelihood (topic assignment and word observation components of ELBO) for best candidate to Lbound
            else: # TODO: not test the else part yet
                # selecting subsequent node, some active nodes are already selected
                temp = np.zeros(tot_tops)  # indices of active nodes (K x 1)
                temp[idx_active] = np.transpose((np.arange(0, len(idx_active))))
                idx_clps = temp[idx_pick].astype(int)  # indices of selected nodes
                num_act = len(idx_active)  # number of active nodes
                vec = np.max(penalty[idx_clps, :], axis = 0)  # word-wise max scaled Elogtheta + Elogpi (1 x Wd)

                # remove scaled Elogtheta + Elogpi for *best* node selected so far
                # from scaled Elogtheta + Elogpi for *each* node selected so far
                # and exponentiate (compute unnormalized nu, adjusting to prevent under/overflow) (Ka x Wd)
                C_act = C_act - np.matlib.repmat(vec, num_act, 1)
                C_act = np.exp(C_act)

                # this part is a little tricky...
                # note below that we set score = -inf for selected nodes, so in the matrix arithmetic that leads up to that,
                # focus on the rows corresponding to active but not selected nodes.  we find that (for one of those rows)
                # numerator is Elogtheta + Elogpi for the selected nodes and the unselected node corresponding to the
                # current row, weighted by unnormalized nu (C_act) and the word counts (which appear in penalty);
                numerator = np.multiply(C_act, penalty)  # scaled Elogtheta + Elogpi, scaled by nu (Ka x Wd)
                numerator = numerator + np.matlib.repmat(np.sum(numerator[idx_clps, :], 0), num_act, 1)
                # denominator is the normalizer for nu for the selected nodes and the unselected node corresponding to
                # the given row;
                denominator = C_act + np.matlib.repmat(np.sum(C_act[idx_clps, :], 0), num_act, 1)
                # vec is sum of - nu' log nu' across the selected nodes;
                vec = np.sum(np.multiply(C_act[idx_clps, :], np.log(eps + C_act[idx_clps, :])), 0)
                # H is - nu log nu
                H = np.log(denominator) - (np.multiply(C_act, np.log(C_act + eps)) + np.matlib.repmat(vec.reshape(1,-1), num_act, 1)) / denominator # TODO: confirm same with element wise division
                # and score is per-topic weighted sum of nu * (Elogtheta + Elogpi) (weighted by word counts)
                # + Elogp - per-topic weighted sum of nu log nu (weighted by word counts)
                score = np.sum(np.divide(numerator, denominator), axis=1) + ElnPtop_act + np.dot(H, Xcnt[Xid_keys[d]])
                score[idx_clps] = - np.inf  # set score of selected nodes to -inf (little hack)
                temp= np.nanmax(score)  # compute best candidate (active but not selected nodes)
                #idx_this = np.argmax(score) # TODO: select different from matlab, python chose the nan not the -inf
                idx_this = np.argwhere(score == temp).reshape(-1)[0]
                idx_pick.append(idx_active[idx_this])  # store globl index of best candidate # TODO: to use the append
                Lbound.append(temp - ElnPtop_act[idx_this])  # append likelihood (topic assignment and word observation components of ELBO) for best candidate to Lbound

            #  update candidates according to new selected node
            idx_this = np.argwhere(id_parent == id_parent[idx_pick[-1]]).reshape(-1)  # find global indices of recently selected node's siblings
            t1, t2, _ = np.intersect1d(idx_this, idx_pick, return_indices=True) # TODO: confirm right
            #t1, t2 = idx_this.intersection(idx_pick)  # remove nodes already selected (including recently selected node) from siblings (idx_this), leaving unselected siblings remaining
            idx_this = np.delete(idx_this, t2) # TODO: confirm delete the value or index
            vec_DPweights[idx_this] = vec_DPweights[idx_this] + Eln1_V  # add local Elog(1-V) prior to unselected sibling DP Elogpi (excludes level penalty)
            ElnP_d[idx_this] = ElnP_d[idx_this] + Eln1_V  # update local Elogpi with Elog(1-V) prior for unselected siblings
            idx_add = np.argwhere(id_parent == id_me[idx_pick[-1]]).reshape(-1)  # find global indices of recently selected node's children
            vec_DPweights[idx_add] = ElnV  # add local ElogV prior to new children DP Elogpi (excludes level penalty)
            ElnP_d[idx_add] = ElnV + level_penalty[idx_add]  # add local ElogV and ElogU... prior to new children Elogpi

            # walk up tree from recently-selected node, adding DP Elogpi (excludes level penalty) to new-children Elogpi... what about new-children DP Elogpi (excludes level penalty)?
            bool2 = 1
            idx = idx_pick[-1]
            while bool2:
                # if id_me[idx] ~= np.log(2):
                # if not idx.size == 0:
                if idx.size == 1:  # id_me(idx) ~= log(2) # TODO: confirm this condition
                    ElnP_d[idx_add] = ElnP_d[idx_add] + vec_DPweights[idx]
                    idx = np.argwhere(id_me == id_parent[idx]).reshape(-1)
                else:
                    bool2 = 0

            # stop if relative change in ELBO is less than 1e-3 or subtree has 20 nodes
            if len(Lbound) > 1:
                if np.abs(Lbound[-1] - Lbound[-2]) / np.abs(Lbound[-2]) < 10 ** - 3 or len(Lbound) == 20:
                    bool = 0
            hist_levels[len(Tree['me'][idx_pick[-1]]) - 1] = hist_levels[len(Tree['me'][idx_pick[-1]]) - 1] + 1
            #plot(Lbound); title(num2str(length(Tree(idx_pick(end)).me))); pause(.1);

        size_subtree[d] = len(idx_pick)  # store size of selected subtree
        # learn document parameters for subtree
        T = len(idx_pick)  # again, size of subtree
        ElnB_d = ElnB[np.ix_(idx_pick, Xid[Xid_keys[d]])]  # Elogtheta for given subtree, words #subarray
        ElnP_d = 0 * ElnP_d[idx_pick] - 1  # Elogpi for given subtree... ignored for first iteration
        cnt_old = np.zeros((len(idx_pick), 1)) # TODO: confirm if need to change

        bool_this = 1
        num = 0
        ElnP_d = ElnP_d.reshape(-1, 1)
        while bool_this:

            num = num + 1
            # estimate nu
            C_d = ElnB_d + np.matlib.repmat(ElnP_d, 1, len(Xid[Xid_keys[d]]))
            C_d = C_d - np.matlib.repmat(np.max(C_d, axis = 0), T, 1)
            C_d = np.exp(C_d)
            C_d = C_d / np.matlib.repmat(np.sum(C_d, axis = 0), T, 1)
            # store nu sums (one per topic) in cnt
            cnt = np.dot(C_d, Xcnt[Xid_keys[d]].reshape(-1,1))
            # estimate Elogpi
            ElnP_d = func_doc_weight_up(cnt, id_parent[idx_pick], gamma2, gamma3, gamma4, Tree_mat[np.ix_(idx_pick, idx_pick)])
            # stop if rel change in nu sums is less than 1e-3 or 50 iters elapsed
            if sum(np.abs(cnt - cnt_old)) / sum(cnt) < 10 ** - 3 or num == 50:
                bool_this = 0
            cnt_old = cnt
            #         stem(cnt); title(num2str(num)); pause(.1);

        #  update batch theta ss
        B_up[np.ix_(idx_pick, Xid[Xid_keys[d]])] = B_up[np.ix_(idx_pick, Xid[Xid_keys[d]])] + \
                                           np.multiply(C_d, np.matlib.repmat(Xcnt[Xid_keys[d]].reshape(1,-1), len(idx_pick), 1))
        # TODO: confirm the difference between multiply, and dot
        # update batch V ss
        weight_up[idx_pick] = weight_up[idx_pick] + 1

    # plt.stem(np.sum(B_up, 1), use_line_collection=True) # TODO: confirm if it is right
    # plt.show()

    # #plt.bar(np.histogram(size_subtree, np.arange(0,20)))
    # #plt.hist(size_subtree, np.arange(0, 20), density=True,histtype='bar',)
    # plt.bar(np.arange(20), size_subtree)
    # plt.show()
    # plt.bar(np.arange(len(hist_levels)), hist_levels / D)
    # plt.show()

    # update tree
    # M-step
    # note scale here is as used in init, e.g. 100D/K... (?!)
    # and note the D variable below is the *batch* size
    for i in range(tot_tops):
        if rho == 1: # i =0
            Tree['lambda_sums'][i] = scale * B_up[i, :] / D
        else: # i = else
            # compute avg theta ss for this topic
            vec = np.ones(B_up.shape[1])
            vec = vec / sum(vec)
            vec = sum(B_up[i, :]) * vec
            # set theta ss to (1 - rho) * old + rho * ((1 - rho/10) * new + rho/10 * avg theta ss for this topic)
            Tree['lambda_sums'][i] = (1 - rho) * Tree['lambda_sums'][i] + rho * ((1 - rho / 10) * scale * B_up[i, :] / D
                                                                           + (rho / 10) * scale * vec / D)
            Tree['tau_sums'][i] = (1 - rho) * Tree['tau_sums'][i] + rho * scale * weight_up[i] / D

        return Tree