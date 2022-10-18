import numpy as np
import pandas as pd



print("Which data:\n0:\tmovielens\n1:\tamazon\n")
data = input("")
if data == "0":
    N1 = 5950
    N2 = 2811
    e = pd.read_csv("data/edges_movies.csv", header = None).to_numpy()
    test = []
    for i in range(N1):
        test.append(e[np.where(e[:,0]==i)][np.random.randint(e[np.where(e[:,0]==i)].shape[0])])
    np.savetxt("test_set_movies.csv", np.array(test), delimiter = ",", fmt = "%d")
else:
    N1 = 4881
    N2 = 2686
    e = pd.read_csv("data/edges_amazon.csv", header = None).to_numpy()
    test = []
    for i in range(N1):
        test.append(e[np.where(e[:,0]==i)][np.random.randint(e[np.where(e[:,0]==i)].shape[0])])
    np.savetxt("test_set_amazon.csv", np.array(test), delimiter = ",", fmt = "%d")