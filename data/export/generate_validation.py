import numpy as np
import math

f = open('train.txt', 'r')
lines = f.readlines()
lines = [l.strip('\n\r') for l in lines]
lines = [l.split(' ') for l in lines]

users = [l[0] for l in lines]
train = [l[1:] for l in lines]
train = [[int(l) for l in a] for a in train]

validation = []
for i in range(len(train)):
    validation.append([])
    check = 0
    while check == 0:
        r = np.random.randint(0,len(train[i]))    
        if train[i][r] not in validation[i]:
            validation[i].append(train[i][r])
            if len(validation[i]) == math.ceil(len(train[i])/10):
                check = 1
            

f = open("validation.txt", "w+")
for i in range(len(validation)):
    f.write("{}".format(i))
    for j in range(len(validation[i])):
        f.write(" {}".format(validation[i][j]))
    f.write("\n")
f.close()