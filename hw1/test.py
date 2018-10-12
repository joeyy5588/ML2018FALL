import numpy as np
import pandas as pd
import sys
import csv
data = pd.read_csv(sys.argv[1], encoding = 'Big5', header = None).values
w = np.load('w.npy')
mean = np.load('mean.npy')
std = np.load('std.npy')
bias = np.load('bias.npy')
data = np.delete(data,[0,1], 1)
data[ data == 'NR'] = 0
data = data.astype(np.float)
x = []
sol = []

temp = np.hstack((data[:18, :], data[18:36, :]))
#for i in range(12):
for j in range(2, 260):
	temp = np.hstack((temp, data[18*j:18*(j+1), :]))
temp = np.delete(temp, 3, 0)
pmsquare = np.multiply(temp[8], temp[8])
temp = np.vstack((temp, pmsquare))

temp = (temp - mean)/std
for i in range(260):
	t = []
	for j in range(18):
		for k in range(9):
			t.append(temp[j][k + 9 * i])
	x.append(t)
x = np.array(x)
with open(sys.argv[2], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id', 'value'])
    for i in range(260):
    	y_n = np.dot(x[i], w) + bias
    	y = y_n * std[8][0] + mean[8][0]
    	if y < 0:
    		y = 0
    	spamwriter.writerow(['id_' + str(i), y])

