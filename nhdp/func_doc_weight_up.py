import numpy as np
from scipy.special import psi
    
def func_doc_weight_up(cnt=None, id_parent=None, gamma2=None, gamma3=None, gamma4=None, Tree_mat=None):
    # update expected log probability of each topic selected for this document

    T = len(cnt)
    ElnP_d = np.zeros((T,1))
    bin_cnt1 = cnt
    bin_cnt0 = np.dot(Tree_mat, cnt)
    Elnbin1 = psi(bin_cnt1 + gamma3) - psi(bin_cnt1 + bin_cnt0 + gamma3 + gamma4)
    Elnbin0 = psi(bin_cnt0 + gamma4) - psi(bin_cnt1 + bin_cnt0 + gamma3 + gamma4)
    # # don't re-order weights
    # stick_cnt = bin_cnt1+bin_cnt0;
    # partition = np.unique(id_parent);
    # for i = 1:length(partition)
    #     idx = find(id_parent==partition(i));
    #     t1 = stick_cnt(idx);
    #     t3 = rev_cumsum(t1);
    #     if length(t3) > 1
    #         t4 = [t3(2:end) ; 0];
    #         t5 = [0 ; psi(t4(1:end-1)+gamma2) - psi(t1(1:end-1)+t4(1:end-1)+1+gamma2)];
    #     else
    #         t4 = 0;
    #         t5 = 0;
    #     end
    #     ElnP_d(idx) =  psi(t1+1) - psi(t1+t4+1+gamma2) + cumsum(t5);
    # end
    # this = ElnP_d + Elnbin1 + Tree_mat'*(Elnbin0 + ElnP_d);
    # ElnP_d = this;

    # re-order weights
    stick_cnt = bin_cnt1 + bin_cnt0
    stick_cnt = stick_cnt.reshape(-1)
    partition = np.unique(id_parent)
    for i in range(len(partition)):
        idx = np.argwhere(id_parent == partition[i]).reshape(-1) # replace find with np.argwhere()
        t1 = stick_cnt[idx]
        # sort list in-place in descending order

        t1 = sorted(t1, reverse=True)
        idx_sort = np.flip(np.argsort(t1)).reshape(-1) # decending order of index

        # t3 = rev_cumsum(t1) # Todo: find a proper replacement
        #t3 = np.flip(np.cumsum(t1)) # Todo: need to be confirm https://stackoverflow.com/questions/16541618/perform-a-reverse-cumulative-sum-on-a-numpy-array
        t3 = np.cumsum(t1[::-1])[::-1]

        if len(t3) > 1:
            t4 = np.append(t3[1:], 0).reshape(-1,1)
            temp = psi(t4[0:-1] + gamma2) - psi(np.array(t1[0:-1]).reshape(-1,1) + t4[0:-1] + 1 + gamma2)
            t5 = np.insert(temp, 0, [0])
        else:
            t4 = 0
            t5 = 0
        t1 = np.array(t1).reshape(-1,1)
        weights = psi(t1 + 1) - psi(t1 + t4 + 1 + gamma2) + np.cumsum(t5).reshape(-1,1)
        ElnP_d[idx[idx_sort]] = weights

    this = ElnP_d + Elnbin1 + np.dot(Tree_mat,(Elnbin0 + ElnP_d))
    ElnP_d = this
    return ElnP_d