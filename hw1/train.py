import numpy as np
import pandas as pd
import sys
import math
import random
data = pd.read_csv('train.csv', encoding = 'Big5').values
data = np.delete(data, [0,1,2],1)
data[ data == 'NR'] = 0
data = data.astype(np.float)
x = []
y = []

temp = np.hstack((data[:18, :], data[18:36, :]))
for i in range(12):
	if i == 0: 
		for j in range(2, 20):
			temp = np.hstack((temp, data[18*j:18*(j+1), :]))
	elif i != 2 and i != 3 and i != 11:
		for j in range(20):
			temp = np.hstack((temp, data[18*j + i*360:18*(j+1) + i*360, :]))
temp = np.delete(temp, 3, 0)
pmsquare = np.multiply(temp[8], temp[8])
temp = np.vstack((temp, pmsquare))
'''for i in range(12):
	temp1 = temp[:,480*i:480*(i+1)]
	me = np.mean(temp1, axis = 1, keepdims = True)
	s = np.std(temp1, axis = 1, keepdims = True)
	compare_min = np.full(temp1[8].shape, (me[8][0] - 2 * s[8][0]))
	compare_max = np.full(temp1[8].shape, (me[8][0] + 2 * s[8][0]))
	compare_1 = np.greater(temp1[8], compare_max)
	outofrange_1 = np.where(compare_1 == True)[0]
	outofrange_2 = np.where(temp1[8] < 0)[0]
	for idx in outofrange_1:
		temp[8][idx + 480 * i] = me[8][0] + 2 * s[8][0]
	for idx in outofrange_2:
		temp[8][idx + 480 * i] = 0
	compare_2 = np.less(temp[8, 480*i:480*(i+1)], compare_min)
	outofrange_3 = np.where(compare_2 == True)[0]
	for idx in outofrange_3:
		temp[8][idx + 480 * i] = me[8][0] - 2 * s[8][0]'''
'''for i in range(12):
	temp1 = temp[:,480*i:480*(i+1)]
	me = np.mean(temp1, axis = 1, keepdims = True)
	s = np.std(temp1, axis = 1, keepdims = True)
	#temp1 = (temp1-me) / s
	for j in range(18):
		compare_min = np.full(temp1[j].shape, (me[j][0] - 2 * s[j][0]))
		compare_max = np.full(temp1[j].shape, (me[j][0] + 2 * s[j][0]))
		compare_1 = np.greater(temp1[j], compare_max)
		outofrange_1 = np.where(compare_1 == True)[0]
		outofrange_2 = np.where(temp1[j] < 0)[0]
		for idx in outofrange_1:
			temp[j][idx + 480 * i] = me[j][0] + 2 * s[j][0]
		for idx in outofrange_2:
			temp[j][idx + 480 * i] = 0
		compare_2 = np.less(temp[j, 480*i:480*(i+1)], compare_min)
		outofrange_3 = np.where(compare_2 == True)[0]
		for idx in outofrange_3:
			temp[j][idx + 480 * i] = me[j][0] - 2 * s[j][0]'''

me = np.mean(temp, axis = 1, keepdims = True)
s = np.std(temp, axis = 1, keepdims = True)
temp = (temp - me)/s
for i in range(9):
	if i < 9:
		for l in range(471):
			y.append(temp[8][9+l+480*i])
			t = []
			for m in range(18):
				for n in range(9):
					t.append(temp[m][n+l+480*i])
			x.append(t)
x = np.array(x)
y = np.array(y)
'''print(x)
print(y)
print(x.shape)
print(y.shape)'''
lr = 0.0001
lam = 0.0001
beta_1 = np.full(x[0].shape, 0.9)
beta_2 = np.full(x[0].shape, 0.99)
w = np.full(x[0].shape, 0)
m_t = np.full(x[0].shape, 0)
v_t = np.full(x[0].shape, 0)
m_t_b = 0
v_t_b = 0
t = 0
epsilon = 0.00000001
bias = 0

for num in range(1000000):
	if num%1000 == 0 and num >= 1000:
		RMS = 0
		for i in range(len(x)):
			y_n = np.dot(x[i], w) + bias
			valid = y[i]* s[8][0] + me[8][0]
			y_v = y_n * s[8][0] + me[8][0]
			if y_v < 0:
				y_v = 0
			RMS += (valid - y_v) * (valid - y_v)
		RMS /= len(x)
		RMS = math.sqrt(RMS)
		print(RMS)
	t+=1
	idx = random.randint(0, len(x)-1)
	xt = x[idx].transpose()
	loss = y[idx] - np.dot(x[idx],w) - bias 
	g_t = np.dot(xt,loss) * (-2) #+  2 * lam * np.sum(w)
	g_t_b = loss * (-2)
	m_t = beta_1*m_t + (1-beta_1)*g_t 
	v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t) 
	m_cap = m_t/(1-(beta_1**t))
	v_cap = v_t/(1-(beta_2**t))
	m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
	v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b) 
	m_cap_b = m_t_b/(1-(0.9**t))
	v_cap_b = v_t_b/(1-(0.99**t))
	w_0 = np.copy(w)
	w = w - (lr*m_cap)/(np.sqrt(v_cap)+epsilon)
	bias = bias - (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)
'''print(w)
print(y[1])
print(y[1] - np.dot(x[1],w))'''
'''L = []
for num in range(5652):
	L.append(y[num] - np.dot(x[num],w) - bias)
L = np.array(L)
print(L)
print(np.mean(L))'''
barr = np.array(bias)
RMS = 0
for i in range(len(x)):
	y_n = np.dot(x[i], w) + bias
	valid = y[i]* s[8][0] + me[8][0]
	y_v = y_n * s[8][0] + me[8][0]
	if y_v < 0:
		y_v = 0
	RMS += (valid - y_v) * (valid - y_v)
RMS /= len(x)
RMS = math.sqrt(RMS)
print(RMS)
print(np.linalg.norm(w, 2))

np.save('w.npy', w)
np.save('mean.npy', me)
np.save('std.npy', s)
np.save('bias.npy', barr)

