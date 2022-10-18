import numpy as np
import pandas as pd
import similarity_metrics as sml
import sklearn.metrics as metr

print("choose the type of data to use:\n0:\texport data\n1:\tmovielens data\n2:\tamazon data\n")
data = input("")
print("choose the similarity metric:\n0:\tcommon neighbors\n1:\tjaccard index\n2:\tadamic/adar\n3:\tpreferential attachment\n4:\tresource allocation\n5:\tcosine similarity\n6:\tSorensen index\n7:\thub depressed index\n8:\thub promoted index\n9:\ttaxonomy network\n10:\tprobabilistic spreading\n11:\tsapling similarity\n")
similarity = input("")
print("project into the layer of:\n0:\tusers\n1:\titems\n")
p = int(input(""))
if p == 0:
    project = "users"
else:
    project = "items"
    
print(p,project)
if data == "0":
    N1 = 169
    N2 = 5040
    # LOADING EXPORT MATRICES AND COMPUTING REVEALED COMPARATIVE ADVANTAGE    
    RCA = pd.read_csv("data/RCA/RCA1996_2013.csv", header = None).to_numpy()
    M = (RCA>=1).astype(np.int_)
    M_test = (pd.read_csv("data/RCA/RCA2018.csv", header = None).to_numpy()>=1).astype(np.int_)
    M_input_test = (pd.read_csv("data/RCA/RCA2013.csv", header = None).to_numpy()>=1).astype(np.int_)
    activations = (M_input_test<1).astype(bool)
elif data == "1":
    N1 = 5950
    N2 = 2811
    # LOADING USERS-MOVIES EDGES AND REMOVING ONE MOVIE FOR EACH USER
    e = pd.read_csv("data/edges_movies.csv", header = None).to_numpy()
    test = pd.read_csv("test_set_movies.csv", header = None).to_numpy()
    M = np.zeros([N1,N2])
    M_test = np.zeros([N1,N2])
    for i in range(e.shape[0]):
        if e[i,1] != test[e[i,0]][1]:
            M[e[i,0],e[i,1]] = 1
        else:
            M_test[e[i,0],e[i,1]] = 1
    M_input_test = M
    activations = (M<1).astype(bool)
    
elif data == "2":
    N1 = 4881
    N2 = 2686
    # LOADING USERS-PRODUCTS EDGES AND REMOVING ONE PRODUCT FOR EACH USER
    e = pd.read_csv("data/edges_amazon.csv", header = None).to_numpy()
    test = pd.read_csv("test_set_amazon.csv", header = None).to_numpy()
    M = np.zeros([N1,N2])
    M_test = np.zeros([N1,N2])
    for i in range(e.shape[0]):
        if e[i,1] != test[e[i,0]][1]:
            M[e[i,0],e[i,1]] = 1
        else:
            M_test[e[i,0],e[i,1]] = 1
    M_input_test = M
    activations = (M<1).astype(bool)
    
if similarity == "0":
    B = sml.common_neighbors(M,p)
    m = "CN"
elif similarity == "1":
    B = sml.jaccard(M,p)
    m = "JA"
elif similarity == "2":
    B = sml.adamic_adar(M,p)
    m = "AD"
elif similarity == "3":
    B = sml.preferential_attachment(M,p)
    m = "PA"
elif similarity == "4":
    B = sml.resource_allocation(M,p)
    m = "RA"
elif similarity == "5":
    B = sml.cosine_similarity(M,p)
    m = "CS"
elif similarity == "6":
    B = sml.sorensen(M,p)
    m = "SO"
elif similarity == "7":
    B = sml.hub_depressed_index(M,p)
    m = "HD"
elif similarity == "8":
    B = sml.hub_promoted_index(M,p)
    m = "HP"
elif similarity == "9":
    B = sml.taxonomy_network(M,p)
    m = "TN"
elif similarity == "10":
    B = sml.probabilistic_spreading(M,p)
    m = "PS"
elif similarity == "11":
    B = sml.sapling(M,p)
    m = "SAP"
np.savetxt("similarities/B_{}_{}.csv".format(m,project), B, delimiter = ",", fmt = "%5f")

if data == "0":
    if p == 0:
        S = np.nan_to_num((np.dot(B,M_input_test).T/np.sum(abs(B), axis = 1)).T)
    else:
        S = np.nan_to_num((np.dot(B, M_input_test.T).T/np.sum(abs(B), axis = 1)))
    AUCPR = metr.average_precision_score(M_test.flatten()[activations.flatten()],S.flatten()[activations.flatten()])
    mP_10 = 0.0
    P_1000 = np.sum(M_test.flatten()[activations.flatten()][S.flatten()[activations.flatten()].argsort()][-1000:])/1000
    MAP = 0
    c10 = 0
    for c in range(S.shape[0]):
        act = activations[c]
        true = M_test[c][act]
        predict = S[c][act]
        MAP += metr.average_precision_score(true, predict)
        if np.sum(true) >= 10:
            c10 += 1
            mP_10 += np.sum(true[predict.argsort()][-10:])/10
    mP_10 = mP_10/c10
    MAP = MAP/S.shape[0]
    print("AUC-PR\tMAP\tprec@1000\tmP@10\n{}\t{}\t{}\t{}".format(AUCPR,MAP,P_1000,mP_10))
elif data == "1" or data == "2":   
    if p == 0:
        S =  np.nan_to_num((np.dot(B,M_input_test).T/np.sum(abs(B),axis = 1)).T)
    else:
        S = np.nan_to_num((np.dot(B, M_input_test.T).T/np.sum(abs(B), axis = 1)))
    AUCPR = metr.average_precision_score(M_test.flatten()[activations.flatten()],S.flatten()[activations.flatten()])
    MAP = 0
    HR5 = 0
    for c in range(S.shape[0]):
        act = activations[c]
        true = M_test[c][act]
        predict = S[c][act]
        MAP += metr.average_precision_score(true, predict)
        if 1 in true[np.flip(predict.argsort())[:5]]:
            HR5 += 1/S.shape[0]
    MAP = MAP/S.shape[0]
    print("AUC-PR\tMAP\tHR@5\n{}\t{}\t{}".format(AUCPR,MAP,HR5))
    

