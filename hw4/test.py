import numpy as np
import pandas as pd
import csv
import jieba
import sys
import re
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import torch.optim as optim
import torch
import torch.nn as nn
from model import RNN
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

use_cuda = torch.cuda.is_available()
embedding_model = Word2Vec.load("word2vec256.model")
word_vectors = embedding_model.wv
del embedding_model

def preprocessing():
	jieba.load_userdict(sys.argv[2])
	data = 0
	input_dim = 100
	with open(sys.argv[1], encoding = 'utf8') as f:
		content = f.readlines()
		data = [x.strip('\n') for x in content]
		data = [x[x.find(',')+1 : ] for x in data]
	seg_list = [jieba.lcut(x) for x in data]
	seg_list = seg_list[1:][:]
	for i in range(len(seg_list)):
		while len(seg_list[i]) < input_dim:
			seg_list[i].append('.')
		seg_list[i] = seg_list[i][:input_dim]
	#seg_list = seg_list[:][:input_dim]
	return seg_list

test_data = preprocessing()

class MyDataset(Dataset):

	def __init__(self, data, transform=None):
		self.data = data
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		wordidx = []
		for x in self.data[index]:
			try:
				wordidx.append(word_vectors.vocab[x].index)
			except:
				wordidx.append(word_vectors.vocab['.'].index)
		wordidx = np.array(wordidx)
		label = index
		return wordidx, label

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
net = RNN(word_vectors.vectors.shape[0], word_vectors.vectors.shape[1], HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, word_vectors.vectors)
if use_cuda:
	net = net.cuda()
filename = ('RNN.pt')
net.load_state_dict(torch.load(filename))#vgg16_1.pt
net.eval()
test_dataset = MyDataset(test_data, sys.argv[1])
test_loader = DataLoader(test_dataset, batch_size = 1)


def test():
	with open(sys.argv[3], 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['id', 'label'])
		with torch.no_grad():
			for i, (images, labels) in enumerate(test_loader):
				if use_cuda:
					images, labels = images.cuda(), labels.cuda()
				predictions = net(images).data#.squeeze(1)
				rounded_preds = torch.round(torch.sigmoid(predictions))
				#if i%100 == 0: print(i, rounded_preds.data.item())
				spamwriter.writerow([labels.data.item(), str(int(rounded_preds.data.item()))])

test()
