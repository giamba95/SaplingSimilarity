import numpy as np

def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data,r,k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def read_data(data):
    if data == "amazon-product":
        Nu = 6121
        Ni = 2744
    elif data == "export":
        Nu = 169
        Ni = 5040
    elif data == "gowalla":
        Nu = 29858
        Ni = 40981
    elif data == "yelp2018":
        Nu = 31668
        Ni = 38048
    else:
        Nu = 52643
        Ni = 91599
    f = open('../data/{}/train.txt'.format(data), 'r')
    lines = f.readlines()
    lines = [l.strip('\n\r') for l in lines]
    lines = [l.split(' ') for l in lines]
    train = [l[1:] for l in lines]
    for i in range(len(train)):
        if train[i] == [""]:
            train[i] = []
    train = [[int(x) for x in i] for i in train]


    f = open('../data/{}/test.txt'.format(data), 'r')
    lines = f.readlines()
    lines = [l.strip('\n\r') for l in lines]
    lines = [l.split(' ') for l in lines]
    test = [l[1:] for l in lines]
    for i in range(len(test)):
        if test[i] == [""]:
            test[i] = []
    test = [[int(x) for x in i] for i in test]
    
    M_train = np.zeros([Nu,Ni])
    for u in range(Nu):
        if len(train[u]) != 0:
            for i in range(len(train[u])):
                M_train[u,int(train[u][i])] = 1
                
    return Nu, Ni, M_train, train, test

def read_data_validation(data):
    if data == "amazon-product":
        Nu = 6121
        Ni = 2744
    elif data == "export":
        Nu = 169
        Ni = 5040
    elif data == "gowalla":
        Nu = 29858
        Ni = 40981
    elif data == "yelp2018":
        Nu = 31668
        Ni = 38048
    else:
        Nu = 52643
        Ni = 91599
    f = open('../data/{}/train.txt'.format(data), 'r')
    lines = f.readlines()
    lines = [l.strip('\n\r') for l in lines]
    lines = [l.split(' ') for l in lines]
    train = [l[1:] for l in lines]
    for i in range(len(train)):
        if train[i] == [""]:
            train[i] = []
    train = [[int(x) for x in i] for i in train]


    f = open('../data/{}/validation.txt'.format(data), 'r')
    lines = f.readlines()
    lines = [l.strip('\n\r') for l in lines]
    lines = [l.split(' ') for l in lines]
    validation = [l[1:] for l in lines]
    for i in range(len(validation)):
        if validation[i] == [""]:
            validation[i] = []
    validation = [[int(x) for x in i] for i in validation]
    
    for i in range(len(train)):
        for j in range(len(validation[i])):
            train[i].remove(validation[i][j])
            
    M_train = np.zeros([Nu,Ni])
    for u in range(Nu):
        if len(train[u]) != 0:
            for i in range(len(train[u])):
                M_train[u,int(train[u][i])] = 1
                
    return Nu, Ni, M_train, train, validation


def scores(train, test, rec, Nu, Ni, K):
    ndcgK = 0.0
    recK = 0.0
    precK = 0.0
    R = []
    for u in range(Nu):
        pred = rec[u,:]
        true = np.zeros([Ni])
        true[test[u]] = 1
        pred = np.delete(pred,train[u])
        true = np.delete(true,train[u])
        idx = np.flip(pred.argsort())
        R.append(list(true[idx[:K]]))
        scor = RecallPrecision_ATk(test_data = [test[u]], r = np.array([R[u]]), k = K)
        precK += np.nan_to_num(scor["precision"]/Nu)
        recK += np.nan_to_num(scor["recall"]/Nu)
        ndcgK += np.nan_to_num(NDCGatK_r(test_data = [test[u]], r = np.array([R[u]]), k = K)/Nu)
    return precK, recK, ndcgK

