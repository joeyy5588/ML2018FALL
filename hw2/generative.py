import numpy as np
import pandas as pd
import sys
import math
import csv

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def normalization(temp):
    me = np.mean(temp, axis = 0)
    s = np.std(temp, axis = 0)
    s[s == 0] = 1
    temp = (temp - me)/s
    return temp

train_x = pd.read_csv('train_x.csv', encoding = 'Big5').values.astype(np.float)
train_y = pd.read_csv('train_y.csv', encoding = 'Big5').values.astype(np.float)
test_x = pd.read_csv('test_x.csv', encoding = 'Big5').values.astype(np.float)
#train_x = normalization(train_x)
#test_x = normalization(test_x)
train_x1 = train_x[np.where(train_y == 1)[0], :]
train_x0 = train_x[np.where(train_y == 0)[0], :]
mean_1 = np.mean(train_x1, axis = 0)
mean_0 = np.mean(train_x0, axis = 0)
print(train_x1.shape[0], train_x0.shape[0])
sigma_1 = np.dot((train_x1 - mean_1).transpose(), (train_x1 - mean_1)) / train_x1.shape[0]
sigma_0 = np.dot((train_x0 - mean_0).transpose(), (train_x0 - mean_0)) / train_x0.shape[0]
shared_sigma = (train_x1.shape[0] * sigma_1 + train_x0.shape[0] * sigma_0) / (train_x1.shape[0] + train_x0.shape[0])
sigma_inverse = np.linalg.inv(shared_sigma)
w = np.dot((mean_1 - mean_0), sigma_inverse)
b = (-0.5) * np.dot(np.dot(mean_1.transpose(), sigma_inverse), mean_1) + (0.5) * np.dot(np.dot(mean_0.transpose(), sigma_inverse), mean_0) + np.log(float(train_x1.shape[0])/train_x0.shape[0])
print(w, b)

correct = 0.0
for i in range(20000):
        z = np.dot(train_x[i], w) + b
        y = sigmoid(z)
        threshold = 0.45
        y = np.where(y < threshold, 0, 1)
        if y == train_y[i]:
            correct += 1
print(correct/20000)


with open(sys.argv[1], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id', 'value'])
    for i in range(10000):
        z = np.dot(test_x[i], w) + b
        y = sigmoid(z)
        threshold = 0.45
        y = np.where(y < threshold, 0, 1)
        spamwriter.writerow(['id_' + str(i), y])