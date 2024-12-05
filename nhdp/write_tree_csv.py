import numpy as np
    
def write_tree_csv(Tree = None,filename = None): 
    fh = open(filename,'w')
    fh.write('me,parent,tau_sums,lambda_sums\n' % ())
    for i in np.arange(1,len(Tree)+1).reshape(-1):
        node = Tree(i)
        me = sprintf('%d ',node.me)
        me = me(np.arange(1,end() - 1+1))
        parent = sprintf('%d ',node.parent)
        parent = parent(np.arange(1,end() - 1+1))
        tau_sums = sprintf('%g',node.tau_sums)
        lambda_sums = sprintf('%g ',node.lambda_sums)
        lambda_sums = lambda_sums(np.arange(1,end() - 1+1))
        fh.write('%s,%s,%s,%s\n' % (me,parent,tau_sums,lambda_sums))
    
    fh.close()