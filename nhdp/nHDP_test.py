import numpy as np
import numpy.matlib
from scipy.special import psi
    
def nHDP_test(Xid_test = None,Xcnt_test = None,Xid = None,Xcnt = None,Tree = None,beta0 = None): 
    # Written by John Paisley, jpaisley@berkeley.edu
    
    numtest = 0
    Voc = len(Tree(1).beta_cnt)
    tot_tops = len(Tree)
    D = len(Xid)
    # collects statistics for updating the tree
    B_up = np.zeros((tot_tops,Voc))
    weight_up = np.zeros((tot_tops,1))
    gamma1 = 5
    
    gamma2 = 1
    
    gamma3 = (1 / 3)
    
    gamma4 = (2 / 3)
    
    # put info from Tree struct into matrix and vector form
    ElnB,ElnPtop,id_parent,id_me = func_process_tree(Tree,beta0,gamma1)
    M = np.zeros((tot_tops,Voc))
    cnt_top = np.zeros((tot_tops,1))
    for i in np.arange(1,M.shape[1-1]+1).reshape(-1):
        M[i,:] = Tree(i).beta_cnt + beta0
        M[i,:] = M(i,:) / sum(M(i,:))
        cnt_top[i] = Tree(i).cnt
    
    # Family tree indicator matrix. Tree_mat(j,i) = 1 indicates that node j is
# along the path from node i to the root node
    Tree_mat = np.zeros((tot_tops,tot_tops))
    for i in np.arange(1,tot_tops+1).reshape(-1):
        bool = 1
        idx = i
        while bool:

            idx = find(id_me == id_parent(idx))
            if not len(idx)==0 :
                Tree_mat[idx,i] = 1
            else:
                bool = 0

    
    level_penalty = psi(gamma3) - psi(gamma3 + gamma4) + np.transpose(np.sum(Tree_mat, 1-1)) * (psi(gamma4) - psi(gamma3 + gamma4))
    llik_mean = 0
    temp_vec = np.zeros((1,D))
    N = 0
    all_weights = np.zeros((tot_tops,1))
    # main loop
    for d in np.arange(1,D+1).reshape(-1):
        N = N + sum(Xcnt_test[d])
        ElnB_d = ElnB(:,Xid[d])
        ElnV = psi(1) - psi(1 + gamma2)
        Eln1_V = psi(gamma2) - psi(1 + gamma2)
        ElnP_d = np.zeros((tot_tops,1))
        - inf
        ElnP_d[id_parent == np.log[2]] = ElnV + psi(gamma3) - psi(gamma3 + gamma4)
        #   # select subtree
#     bool = 1;
#     idx_pick = [];
#     Lbound = [];
#     vec_DPweights = zeros(tot_tops,1);                                  # ElnP_d minus the level penalty
#     while bool
#         idx_active = find(ElnP_d > -inf);                                             # index of active (selected and potential) nodes
#         penalty = ElnB_d(idx_active,:) + repmat(ElnP_d(idx_active),1,length(Xid{d}));
#         C_act = penalty;
#         penalty = penalty.*repmat(Xcnt{d},size(penalty,1),1);
#         ElnPtop_act = ElnPtop(idx_active);
#         if isempty(idx_pick)
#             score = sum(penalty,2) + ElnPtop_act;
#             [temp,idx_this] = max(score);
#             idx_pick = idx_active(idx_this);                                          # index of selected nodes
#             Lbound(end+1) = temp - ElnPtop_act(idx_this);
#         else
#             temp = zeros(tot_tops,1);
#             temp(idx_active) = (1:length(idx_active))';
#             idx_clps = temp(idx_pick);                                                # index of selected nodes within active nodes
#             num_act = length(idx_active);
#             vec = max(penalty(idx_clps,:),[],1);
#             C_act = C_act - repmat(vec,num_act,1);
#             C_act = exp(C_act);
#             numerator = C_act.*penalty;
#             numerator = numerator + repmat(sum(numerator(idx_clps,:),1),num_act,1);
#             denominator = C_act + repmat(sum(C_act(idx_clps,:),1),num_act,1);
#             vec = sum(C_act(idx_clps,:).*log(eps+C_act(idx_clps,:)),1);
#             H = log(denominator) - (C_act.*log(C_act+eps) + repmat(vec,num_act,1))./denominator;
#             score = sum(numerator./denominator,2) + ElnPtop_act + H*Xcnt{d}';
#             score(idx_clps) = -inf;
#             [temp,idx_this] = max(score);
#             idx_pick(end+1) = idx_active(idx_this);
#             Lbound(end+1) = temp - ElnPtop_act(idx_this);
#         end
#         idx_this = find(id_parent == id_parent(idx_pick(end)));
#         [t1,t2] = intersect(idx_this,idx_pick);
#         idx_this(t2) = [];
#         vec_DPweights(idx_this) = vec_DPweights(idx_this) + Eln1_V;
#         ElnP_d(idx_this) = ElnP_d(idx_this) + Eln1_V;
#         idx_add = find(id_parent == id_me(idx_pick(end)));
#         vec_DPweights(idx_add) = ElnV;
#         ElnP_d(idx_add) = ElnV + level_penalty(idx_add);
#         bool2 = 1;
#         idx = idx_pick(end);
#         while bool2
#             if ~isempty(idx) #id_me(idx) ~= log(2)
#                 ElnP_d(idx_add) = ElnP_d(idx_add) + vec_DPweights(idx);
#                 idx = find(id_me == id_parent(idx));
#             else
#                 bool2 = 0;
#             end
#         end
#         if length(Lbound) > 5
#             if abs(Lbound(end)-Lbound(end-1))/abs(Lbound(end-1)) < 10^-3 || length(Lbound) == 25
#                 bool = 0;
#             end
#         end
#     end
        idx_pick = np.arange(1,tot_tops+1)
        # learn document parameters for subtree
        T = len(idx_pick)
        ElnB_d = ElnB(idx_pick,Xid[d])
        ElnP_d = 0 * ElnP_d(idx_pick) - 1
        cnt_old = np.zeros((len(idx_pick),1))
        bool_this = 1
        num = 0
        while bool_this:

            num = num + 1
            C_d = ElnB_d + np.matlib.repmat(ElnP_d,1,len(Xid[d]))
            C_d = C_d - np.matlib.repmat(np.amax(C_d,[],1),T,1)
            C_d = np.exp(C_d)
            C_d = C_d / np.matlib.repmat(np.sum(C_d, 1-1),T,1)
            cnt = C_d * np.transpose(Xcnt[d])
            #         ElnP_d = func_doc_weight_up(cnt,id_parent(idx_pick),gamma2,gamma3,gamma4,Tree_mat(idx_pick,idx_pick));
            T = len(cnt)
            ElnP_d = np.zeros((T,1))
            bin_cnt1 = cnt
            bin_cnt0 = Tree_mat * cnt
            Elnbin1 = psi(bin_cnt1 + gamma3) - psi(bin_cnt1 + bin_cnt0 + gamma3 + gamma4)
            Elnbin0 = psi(bin_cnt0 + gamma4) - psi(bin_cnt1 + bin_cnt0 + gamma3 + gamma4)
            stick_cnt = bin_cnt1 + bin_cnt0
            partition = unique(id_parent)
            for i in np.arange(1,len(partition)+1).reshape(-1):
                idx = find(id_parent == partition(i))
                t1 = stick_cnt(idx)
                ElnP_d[idx] = psi(t1 + cnt_top(idx) / sum(cnt_top(idx))) - psi(1 + sum(t1))
            this = ElnP_d + Elnbin1 + np.transpose(Tree_mat) * (Elnbin0 + ElnP_d)
            ElnP_d = this
            if num > 10:
                if sum(np.abs(cnt - cnt_old)) / sum(cnt) < 0.5 * 10 ** - 2 or num == 25:
                    bool_this = 0
            cnt_old = cnt

        Tree_this = Tree_mat(idx_pick,idx_pick)
        id_par_this = id_parent(idx_pick)
        bin_cnt1 = cnt
        bin_cnt0 = Tree_this * cnt
        #     idx = find(bin_cnt0<.01);
        Ebin1 = (bin_cnt1 + gamma3) / (bin_cnt1 + bin_cnt0 + gamma3 + gamma4)
        Ebin0 = (bin_cnt0 + gamma4) / (bin_cnt1 + bin_cnt0 + gamma3 + gamma4)
        #     Ebin1(idx) = 1-eps;
#     Ebin0(idx) = eps;
        Elnbin1 = np.log(Ebin1)
        Elnbin0 = np.log(Ebin0)
        stick_cnt = bin_cnt1 + bin_cnt0
        partition = unique(id_par_this)
        for i in np.arange(1,len(partition)+1).reshape(-1):
            idx = find(id_par_this == partition(i))
            t1 = stick_cnt(idx)
            weights = np.log(t1 + cnt_top(idx) / sum(cnt_top(idx))) - np.log(sum(t1) + 1)
            ElnP_d[idx] = weights
        this = ElnP_d + Elnbin1 + np.transpose(Tree_this) * (Elnbin0 + ElnP_d)
        this = this - np.amax(this)
        P_d = np.exp(this)
        P_d = P_d / sum(P_d)
        vec = np.transpose(P_d) * M(idx_pick,Xid_test[d])
        llik_mean = llik_mean + np.log(vec) * np.transpose(Xcnt_test[d])
        numtest = numtest + sum(Xcnt_test[d])
        #     disp([num2str(d) ' : ' num2str(sum(llik_mean)/(numtest))]);
        #     if mod(d,1000) == 0
#         disp(num2str(llik_mean/N));
#         hold on;
#         temp_vec(d) = llik_mean/N;
#         plot(temp_vec(1:d),'b'); pause(.1);
#         all_weights = all_weights + ElnP_d;
#         stem(all_weights); pause(.1);
#     end
    
    return llik_mean,C_d