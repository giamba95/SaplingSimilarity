import numpy as np
import pandas as pd
import similarities as sml
from parse import parse_args

args = parse_args()
projection = args.projection
if projection == "user-based":
    proj = 0
else:
    proj = 1
model = args.similarity


print("loading data...")
ratings_train = pd.read_csv("data/ratings_train.txt", header = None).to_numpy()
ratings_test = pd.read_csv("data/ratings_test.txt", header = None).to_numpy()

users = np.unique(ratings_train[:,0])
movies = np.unique(ratings_train[:,1])

"""
M_bin element is 1 if a user rated a movie with 3 or higher and 0 otherwise
M element is equal to the rate a user gave to a movie
"""
M_bin = np.zeros([users.shape[0],movies.shape[0]])
M = np.zeros([users.shape[0],movies.shape[0]])
for i in range(ratings_train.shape[0]):
    if ratings_train[i,2] >= 3:
        M_bin[np.where(ratings_train[i,0] == users)[0][0], np.where(ratings_train[i,1] == movies)[0][0]] = 1
    M[np.where(ratings_train[i,0] == users)[0][0], np.where(ratings_train[i,1] == movies)[0][0]] = ratings_train[i,2]



print("measuring the similarity...")
B = sml.similarity(M_bin,model,proj)

print("predicting the ratings...")
predictions = np.zeros(ratings_test.shape[0])
for i in range(ratings_test.shape[0]):
    if proj == 0:
        predictions[i] = np.dot(B[np.where(users == ratings_test[i,0])[0][0],:], M[:,np.where(movies == ratings_test[i,1])[0][0]])/np.sum(abs(B[np.where(users == ratings_test[i,0])[0][0],:][(M[:,np.where(movies == ratings_test[i,1])[0][0]]>0)]))
    else:
        predictions[i] = np.dot(B[np.where(movies == ratings_test[i,1])[0][0],:], M[np.where(users == ratings_test[i,0])[0][0],:])/np.sum(abs(B[np.where(movies == ratings_test[i,1])[0][0],:][(M[np.where(users == ratings_test[i,0])[0][0],:]>0)]))


def ndcg(e_true, ypred):
    users,cu = np.unique(e_true[:,0],return_counts = True)
    nDCG = 0
    for u in range(users.shape[0]):
        true = e_true[np.where(e_true[:,0] == users[u])[0],:][:,2]-1
        pred = ypred[np.where(e_true[:,0] == users[u])[0]]
        true = true[np.flip(pred.argsort())]
        itrue = true[np.flip(true.argsort())]
        DCG = 0
        iDCG = 0
        for i in range(true.shape[0]):
            DCG += (2**true[i]-1)/np.log2((i+1)+1)
            iDCG += (2**itrue[i]-1)/np.log2((i+1)+1)
        if iDCG == 0:
           nDCG += 1
        else:
           nDCG += DCG/iDCG
    return nDCG/users.shape[0]

print("measuring the performance...")
ypred = predictions
ytrue = ratings_test[:,2]
MAE = np.sum(abs(ytrue-np.around(ypred,0)))/ytrue.shape[0]
RMSE = (np.sum((ytrue-np.around(ypred,0))**2)/ytrue.shape[0])**0.5
nDCG = ndcg(ratings_test,ypred)

print("MAE:\t{}\nRMSE:\t{}\nnDCG:\t{}".format(MAE,RMSE,nDCG))
