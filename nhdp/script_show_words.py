# display vocabulary results. the vocabulary is in a cell called vocab.

import numpy as np
ElnB,ElnPtop,id_parent,id_me = func_process_tree(Tree,beta0,5)

idx = 1

idx_p = find(id_me == id_parent(idx))
idx_c = find(id_parent == id_me(idx))
print('*** This node ***')
print(np.array(['Count ',num2str(Tree(idx).cnt)]))
a,b = __builtint__.sorted(Tree(idx).beta_cnt,'descend')
for w in np.arange(1,10+1).reshape(-1):
    print(np.array(['   ',vocab[b(w)]]))

print('*** Parent node ***')
if len(idx_p)==0:
    print('No parent')
else:
    a,b = __builtint__.sorted(Tree(idx_p).beta_cnt,'descend')
    for w in np.arange(1,10+1).reshape(-1):
        print(np.array(['   ',vocab[b(w)]]))

if len(idx_c)==0:
    print('No children')
else:
    for i in np.arange(1,len(idx_c)+1).reshape(-1):
        print(np.array(['Child ',num2str(i),' : Count ',num2str(Tree(idx_c(i)).cnt),' : Index ',num2str(idx_c(i))]))
        a,b = __builtint__.sorted(Tree(idx_c(i)).beta_cnt,'descend')
        for w in np.arange(1,10+1).reshape(-1):
            print(np.array(['   ',vocab[b(w)]]))
