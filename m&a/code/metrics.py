import numpy as np
from sklearn.metrics import precision_recall_curve

def Best_F1(true, predict):
    precision, recall, threshold = precision_recall_curve(true, predict, pos_label = 1)
    if 0 in precision:
        for i in range(np.shape(precision)[0]):
            if precision[i] == 0 and i > 0:
                precision[i] = 0.001
                recall[i] = 0.001
    f1 = 2*precision*recall/(precision+recall)
    return threshold[np.argmax(f1)], np.max(f1)