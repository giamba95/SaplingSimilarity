import numpy as np
import pandas as pd
import metrics as metr
import similarities as sml
from scipy.io import mmread
import sklearn

first_year = 2002
N_years = 11

# modify ex to switch prediction exercise:
# ex = 0: target prediction
# ex = 1: acquirer prediction
# ex = 2: pair prediction
ex = 2

# modify CI to set the class imbalance
CI = 200

# choose the method to use:
# bin = matrix M without weights
# (1) = with the adjustment that accounts for the size of the firm
# (2) = with the adjustment that accounts for the rarity of the technology
S = 7
similarity = ["SS bin","SS(1) bin","SS(2) bin ","SS(1+2) bin","SS","SS(1)","SS(2)","SS(1+2)","LightGCN"]

# loading data
companies = pd.read_csv("../data/companies.txt", header = None).to_numpy()[:,0]
# ma 
# [[target1,acquirer1,year1],[target2,acquirer2,year2],[target3,acquirer3,year3], ... ]
ma = pd.read_csv("../data/ma.txt").to_numpy()
M = []
for y in range(N_years):
    M.append(np.array(mmread('../data/M/M_{}.mtx'.format(first_year+y)).todense()))
N_comp = M[0].shape[0]
N_tech = M[0].shape[1]


# turn co0 to True to consider only firms with 0 co-occurrences
co0 = False
if co0:
    idx = []
    for i in range(ma.shape[0]):
        if np.dot(M[ma[i,2]-first_year][ma[i,0],:],M[ma[i,2]-first_year][ma[i,1],:]) == 0:
            idx.append(i)
    ma = ma[idx]

# computing sapling
B = []
for y in range(N_years):
    print("measuring {}: {}".format(similarity[S],first_year+y))
    if S == 0:
        B.append(sml.sapling_bin(M[y]))
    elif S == 1:
        B.append(sml.sapling_asym_bin(M[y]))
    elif S == 2:
        B.append(sml.sapling_RA_bin(M[y]))
    elif S == 3:
        B.append(sml.sapling_asym_RA_bin(M[y]))
    elif S == 4:
        B.append(sml.sapling(M[y]))
    elif S == 5:
        B.append(sml.sapling_asym(M[y]))
    elif S == 6:
        B.append(sml.sapling_RA(M[y]))
    elif S == 7:
        B.append(sml.sapling_asym_RA(M[y]))
    elif S == 8:
        eu = np.load("../data/lightgcn/embedding_firms_{}.npy".format(first_year+y))
        B.append((np.dot(eu,eu.T)))

       
# target or acquirer prediction
if ex == 0 or ex == 1:
    HR = []
    f1 = []
    MAP = []
    for j in range(20):
        f1_single = []
        MAP_single = []
        HR_single = 0
        print("it:\t{}".format(j),end = "\r")
        for y in range(N_years):
            for i in range(ma[np.where(ma[:,2]==first_year+y)[0]].shape[0]):
                may = ma[np.where(ma[:,2]==first_year+y)[0]]
                predict = []
                true = []
                generate = []
                predict.append(B[y][may[i,0],may[i,1]])
                true.append(1)
                # generate negative samples
                for j in range(CI):
                    check = 0
                    while check == 0:
                        r2 = np.random.randint(0, N_comp)
                        if np.sum(M[y][r2,:]) != 0 and r2 not in generate and r2 != may[i,1] and r2 != may[i,0]:
                            check = 1
                    generate.append(r2)
                    if ex == 0: 
                        predict.append(B[y][r2,may[i,1]]) 
                    elif ex == 1:
                        predict.append(B[y][may[i,0],r2])   
                    true.append(0)
                f1_single.append(metr.Best_F1(np.array(true),np.array(predict))[1])
                MAP_single.append(sklearn.metrics.average_precision_score(np.array(true),np.array(predict)))
                if np.sum(np.array(predict)) != 0:
                    HR_single += np.sum( np.array(true)[np.flip(np.array(predict).argsort())][:5] )            
        HR.append(HR_single)
        MAP.append(np.mean(MAP_single))
        f1.append(np.mean(f1_single))
    f1 = np.array(f1)
    HR = np.array(HR)/ma.shape[0]
    MAP = np.array(MAP)
    print("{},{},{}".format(round(np.mean(f1),4),round(np.mean(MAP),4),round(np.mean(HR),4)))
    

elif ex == 2:
    f1 = []
    AUCPR = []
    prec = []
    for j in range(20):
        print("it:\t{}".format(j),end = "\r")
        predict = []
        true = []
        for y in range(N_years):
            for i in range(ma[np.where(ma[:,2]==first_year+y)[0]].shape[0]):
                may = ma[np.where(ma[:,2]==first_year+y)[0]]
                predict.append(B[y][may[i,0],may[i,1]])
                true.append(1)
            # generate negative samples
            for i in range(CI*ma[np.where(ma[:,2]==first_year+y)[0]].shape[0]):
                check = 0
                while check == 0:
                    r1 = np.random.randint(0,N_comp)
                    r2 = np.random.randint(0,N_comp)
                    if np.sum(M[y][r1,:]) != 0 and np.sum(M[y][r2,:]) != 0:
                        check = 1
                predict.append(B[y][r1,r2])
                true.append(0)
        predict = np.array(predict)
        true = np.array(true)
        f1.append(metr.Best_F1(true,predict)[1])
        prec.append( np.sum( np.array(true)[np.flip(np.array(predict).argsort())][:500] )/500)
        AUCPR.append(sklearn.metrics.average_precision_score(true,predict))
    f1 = np.array(f1)
    AUCPR = np.array(AUCPR)
    prec = np.array(prec)
    print("{},{},{}".format(round(np.mean(f1),4),round(np.mean(AUCPR),4),round(np.mean(prec),4)))

