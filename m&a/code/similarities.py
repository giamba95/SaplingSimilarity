import numpy as np


# bin = matrix M without weights
# RA = resource allocation (each co-occurrence is divided by the 'size' of the technology)
# asym = add the size_of_acquirer/size_of_target factor

def sapling_bin(M):
    M= (M>0).astype(float)
    CO = np.dot(M,M.T)
    N = M.shape[1]
    k = np.max(CO,axis = 1)
    B = np.nan_to_num((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))
    return B

def sapling_asym_bin(M):
    M = (M>0).astype(float)
    kt = np.sqrt(np.sum(M**2,axis = 1))
    CO = np.dot(np.nan_to_num(M),M.T)
    N = np.max(CO)
    k = np.max(CO, axis = 1)
    B = np.nan_to_num(np.nan_to_num(((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1)).T/kt).T*kt)
    return B

def sapling_RA_bin(M):
    M = (M>0).astype(float)
    kt = np.sqrt(np.sum(M**2,axis = 1))
    ki = np.sqrt(np.sum(M**2, axis = 0))
    CO = np.dot(np.nan_to_num(M/ki),M.T)
    N = np.max(CO)
    k = np.max(CO, axis = 1)
    B = np.nan_to_num(np.nan_to_num(((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))))
    return B

def sapling_asym_RA_bin(M):
    M = (M>0).astype(float)
    kt = np.sqrt(np.sum(M**2,axis = 1))
    ki = np.sqrt(np.sum(M**2, axis = 0))
    CO = np.dot(np.nan_to_num(M/ki),M.T)
    N = np.max(CO)
    k = np.max(CO, axis = 1)
    B = np.nan_to_num(np.nan_to_num(((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1)).T/kt).T*kt)
    return B

def sapling(M):
    CO = np.dot(M,M.T)
    N = np.max(CO)
    k = np.max(CO,axis = 1)
    B = np.nan_to_num((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))
    return B

def sapling_asym(M):
    kt = np.sqrt(np.sum(M**2,axis = 1))
    CO = np.dot(np.nan_to_num(M),M.T)
    N = np.max(CO)
    k = np.max(CO, axis = 1)
    B = np.nan_to_num(np.nan_to_num(((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1)).T/kt).T*kt)
    return B

def sapling_asym_RA(M):
    kt = np.sqrt(np.sum(M**2,axis = 1))
    ki = np.sqrt(np.sum(M**2, axis = 0))
    CO = np.dot(np.nan_to_num(M/ki),M.T)
    N = np.max(CO)
    k = np.max(CO, axis = 1)
    B = np.nan_to_num(np.nan_to_num(((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1)).T/kt).T*kt)
    return B

def sapling_RA(M):
    kt = np.sqrt(np.sum(M**2,axis = 1))
    ki = np.sqrt(np.sum(M**2, axis = 0))
    CO = np.dot(np.nan_to_num(M/ki),M.T)
    N = np.max(CO)
    k = np.max(CO, axis = 1)
    B = np.nan_to_num(np.nan_to_num(((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))))
    return B


