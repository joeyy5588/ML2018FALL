import numpy as np
import pandas as pd
import sys
import math
import csv

train_x = pd.read_csv('train_x.csv', encoding = 'Big5')#.values.astype(np.float)
train_y = pd.read_csv('train_y.csv', encoding = 'Big5').values.astype(np.float)
test_x = pd.read_csv('test_x.csv', encoding = 'Big5')#.values.astype(np.float)


def onehotencoding(train_x):
	df = pd.get_dummies(train_x['SEX'])
	df2 = pd.get_dummies(train_x['EDUCATION'])
	df3 = pd.get_dummies(train_x["MARRIAGE"])
	df4 = pd.get_dummies(train_x["PAY_0"])
	df9 = pd.get_dummies(train_x["PAY_6"])
	df5 = pd.get_dummies(train_x["PAY_2"])
	df6 = pd.get_dummies(train_x["PAY_3"])
	df7 = pd.get_dummies(train_x["PAY_4"])
	df8 = pd.get_dummies(train_x["PAY_5"])
	train_x = pd.concat([train_x, df, df2, df3, df4, df5, df6, df7, df8], axis = 1).values.astype(np.float)
	mylist = list(range(11, train_x.shape[1]))
	mylist.insert(0, 0)#decide whether to add age
	#mylist.insert(1, 4)
	train_x = train_x[:, mylist]
	return train_x

def normalization(temp):
	me = np.mean(temp, axis = 0)
	s = np.std(temp, axis = 0)
	s[s == 0] = 1
	temp = (temp - me)/s
	return temp

train_x = onehotencoding(train_x)
test_x = onehotencoding(test_x)
N = np.zeros(test_x.shape[0])
test_x = np.c_[test_x, N]
train_x = normalization(train_x)
test_x = normalization(test_x)
print(train_x.shape, test_x.shape)

w = np.full(train_x[0].shape, 0)
bias = 0
lr = 0.001
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def cross_entropy(train_x, train_y, w):
	#Y = np.clip(Y, 0.00000000000001, 0.99999999999999)
	Y = train_y.reshape(-1)
	Z = np.dot(train_x, w) + bias
	Yhat = sigmoid(Z)
	y = np.clip(Yhat, 0.00000000000001, 0.99999999999999)
	return -(np.sum(Y*np.log(y) + (1 - Y)*np.log(1-y)) )
def logistic_regression(batch, epoch, w, bias, threshold):
    for num in range(epoch):
        randidx = np.random.randint(len(train_x), size = batch)
        # print('randidx', randidx.shape)
        randx = train_x[randidx, :]
        randy = train_y[randidx, :]
        yhat = randy.reshape(-1)
        # print('randx', randx.shape)
        # print('randy', randy.shape)
        # print('w', w.shape)
        # print('w.reshape', w.reshape(-1, 1).shape)
        z = np.dot(randx, w.transpose()) + bias
        # 32 x 79
        #z = randx @ w.reshape(-1, 1) + bias
        # print('z', z.shape)
        y = sigmoid(z)
        #error = cross_entropy(train_x, train_y, w)
        #print(error)
        # input()
        w_grad = np.mean(-1 * randx * (yhat - y).reshape(batch, 1), axis = 0)
        w = w - lr * w_grad
        b_grad = np.mean(-1 * (yhat - y))
        bias = bias - lr * b_grad
    return (w, bias)

w, bias = logistic_regression(32, 20000, w, bias, 0.9)
print(w, bias)
correct = 0.0
for i in range(20000):
        z = np.dot(train_x[i], w) + bias
        y = sigmoid(z)
        threshold = 0.5
        print(z)
        y = np.where(y < threshold, 0, 1)
        if y == train_y[i]:
            correct += 1
print(correct/20000)

with open(sys.argv[1], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id', 'value'])
    for i in range(10000):
        z = np.dot(test_x[i], w) + bias
        y = sigmoid(z)
        threshold = 0.5
        y = np.where(y < threshold, 0, 1)
        spamwriter.writerow(['id_' + str(i), y])
