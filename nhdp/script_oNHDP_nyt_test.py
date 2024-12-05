import numpy as np
cd('../nyt')
scipy.io.loadmat('mult_test')
cd('../oNHDP')
for i in np.arange(1,len(Xid_test)+1).reshape(-1):
    Xid_test[i] = Xid_test[i] + 1

D = len(Xid_test)
perctest = 0.1
Xid = Xid_test
Xcnt = Xcnt_test
Xcnt_test = cell(1,D)
Xid_test = cell(1,D)
for d in np.arange(1,D+1).reshape(-1):
    numW = sum(Xcnt[d])
    numTest = int(np.floor(perctest * numW))
    a,b = __builtint__.sorted(np.random.rand(1,numW))
    wordVec = []
    for i in np.arange(1,len(Xid[d])+1).reshape(-1):
        wordVec = np.array([wordVec,Xid[d](i) * np.ones((1,Xcnt[d](i)))])
    wordTestVec = wordVec(b(np.arange(1,numTest+1)))
    wordTrainVec = wordVec(b(np.arange(numTest + 1,end()+1)))
    Xid[d] = unique(wordTrainVec)
    Xcnt[d] = histc(wordTrainVec,Xid[d])
    Xid_test[d] = unique(wordTestVec)
    Xcnt_test[d] = histc(wordTestVec,Xid_test[d])

num_test = 0
for i in np.arange(1,len(Xcnt_test)+1).reshape(-1):
    num_test = num_test + sum(Xcnt_test[i])

results_mean = np.zeros((1,360))
for i in np.arange(270,360+10,10).reshape(-1):
    tic
    cd('stored_nyt')
    scipy.io.loadmat(np.array(['nHDP_step_nyt_',num2str(i),'.mat']))
    #cd ..
    llik_mean,C_d = nHDP_test(Xid_test,Xcnt_test,Xid,Xcnt,Tree,0.1)
    results_mean[i] = sum(llik_mean) / num_test
    save('oNHDP_nyt_test','results_mean')
    print(np.array(['Finished ',num2str(i),' : ',num2str(toc / 60)]))
