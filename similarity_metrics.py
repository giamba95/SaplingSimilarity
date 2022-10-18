import numpy as np

def common_neighbors(M,projection):
    if projection == 0:
        B = np.dot(M,M.T)
    else:
        B = np.dot(M.T,M)
    return B

def jaccard(M,projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        B = np.nan_to_num(coo/(np.subtract.outer(k, -k)-coo))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        B = np.nan_to_num(coo/(np.subtract.outer(k, -k)-coo))
    return B

def adamic_adar(M,projection):
    if projection == 0:
        k = np.sum(M, axis = 0)
        k = np.nan_to_num(1/np.log10(k))
        B = np.nan_to_num((M*k).dot(M.T))
    else:
        k = np.sum(M, axis = 1)
        k = np.nan_to_num(1/np.log10(k))
        B = np.nan_to_num((M.T*k).dot(M))
    return B

def preferential_attachment(M,projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        B = np.nan_to_num(np.multiply.outer(k, k))
    else:
        k = np.sum(M, axis = 0)
        B = np.nan_to_num(np.multiply.outer(k, k))
    return B

def resource_allocation(M,projection):
    if projection == 0:
        k = np.sum(M, axis = 0)
        B = np.nan_to_num(np.nan_to_num(M/k).dot(M.T))
    else:
        k = np.sum(M, axis = 1)
        B = np.nan_to_num(np.nan_to_num(M.T/k).dot(M))
    return B


def cosine_similarity(M,projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        B = np.nan_to_num(coo/(np.multiply.outer(k,k)**0.5))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        B = np.nan_to_num(coo/(np.multiply.outer(k,k)**0.5))
    return B

def sorensen(M,projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        B = np.nan_to_num(2*coo/(np.subtract.outer(k, -k)))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        B = np.nan_to_num(2*coo/(np.subtract.outer(k, -k)))
    return B


def hub_depressed_index(M,projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        B = np.nan_to_num(coo/np.maximum.outer(k.T,k))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        B = np.nan_to_num(coo/np.maximum.outer(k.T,k))
    return B

def hub_promoted_index(M,projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        B = np.nan_to_num(coo/np.minimum.outer(k.T,k))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        B = np.nan_to_num(coo/np.minimum.outer(k.T,k))
    return B

def taxonomy_network(M,projection):
    if projection == 0:
        k1 = np.sum(M, axis = 1)
        k2 = np.sum(M, axis = 0)
        B = np.nan_to_num(np.nan_to_num(M/k2).dot(M.T)/np.maximum.outer(k1.T,k1))
    else:
        k2 = np.sum(M, axis = 1)
        k1 = np.sum(M, axis = 0)
        B = np.nan_to_num(np.nan_to_num(M.T/k2).dot(M)/np.maximum.outer(k1.T,k1))
    return B

def probabilistic_spreading(M,projection):
    if projection == 0:
        k1 = np.sum(M, axis = 1)
        k2 = np.sum(M, axis = 0)
        B = np.nan_to_num(np.nan_to_num(M/k2).dot(M.T)/k1)
    else:
        k2 = np.sum(M, axis = 1)
        k1 = np.sum(M, axis = 0)
        B = np.nan_to_num(np.nan_to_num(M.T/k2).dot(M)/k1)
    return B


def sapling(M,projection):
    if projection == 0:
        N = M.shape[1]
        k = np.sum(M, axis = 1)
        CO = np.dot(M,M.T)
        B=np.nan_to_num((1-(CO*(1-CO/k)+(k-CO).T*(1-(k-CO).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))
    else:
        N = M.shape[0]
        k = np.sum(M, axis = 0)
        CO = np.dot(M.T,M)
        B=np.nan_to_num((1-(CO*(1-CO/k)+(k-CO).T*(1-(k-CO).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))
    return B