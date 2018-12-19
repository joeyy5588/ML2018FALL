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


def preprocessing():
	jieba.load_userdict(sys.argv[4])
	data = []
	input_dim = 100
	dummy_list = ['.'] * input_dim
	with open(sys.argv[1], encoding = 'utf8') as f:
		content = f.readlines()
		data = [x.strip('\n') for x in content]
		data = [x[x.find(',')+1 : ] for x in data]
		#data = [re.findall(r"[\u4e00-\u9fa5]+", x) for x in content]
		#data = [''.join(x) for x in data]
	seg_list = [jieba.lcut(x) for x in data]
	seg_list = seg_list[1:][:]
	for i in range(len(seg_list)):
		while len(seg_list[i]) < input_dim:
			seg_list[i].append('.')
		seg_list[i] = seg_list[i][:input_dim]
	#seg_list = seg_list[:][:input_dim]
	vec_size = 256
	model = Word2Vec(seg_list, size = vec_size, window = 5, min_count = 1, workers = 4)
	model.save("word2vec" + str(vec_size)+ ".model")
	input_dim = model.wv.vectors.shape[0]
	return seg_list, model, vec_size, input_dim

train_data, embedding_model, embedding_dim, input_dim = preprocessing()
word_vectors = embedding_model.wv
del embedding_model

class MyDataset(Dataset):

	def __init__(self, data, file_path, transform=None):
		self.data = data #np.array(data)
		self.label_list = pd.read_csv(file_path).iloc[:, 1]
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		wordidx = [word_vectors.vocab[x].index for x in self.data[index]]
		label = self.label_list[index]
		label = np.float32(label)
		wordidx = np.array(wordidx)
		return wordidx, label

train_dataset = MyDataset(train_data, sys.argv[2])
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
batch_size = 256
sp = int(np.floor(dataset_size*0.8))
train_indices, val_indices = indices[:sp], indices[sp:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler)
valid_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = valid_sampler)

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
net = RNN(input_dim, embedding_dim, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, word_vectors.vectors)
optimizer = optim.Adam(net.parameters(), lr = 1e-4)
criterion = nn.BCEWithLogitsLoss()
if use_cuda:
	net = net.cuda()
	criterion = criterion.cuda()

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc

def train():
    
    epoch_loss = 0
    epoch_acc = 0
    
    net.train()
    
    for i, (images, labels) in enumerate(train_loader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        
        predictions = net(images).squeeze(1)
        loss = criterion(predictions, labels)
        
        acc = binary_accuracy(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item() * batch_size
        epoch_acc += acc.item() * batch_size
    
    return epoch_loss / sp, epoch_acc / sp
def evaluate():
    
    epoch_loss = 0
    epoch_acc = 0
    
    net.eval()
    
    with torch.no_grad():
    
        for i, (images, labels) in enumerate(valid_loader):
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            predictions = net(images).squeeze(1)
            
            loss = criterion(predictions, labels)
            
            acc = binary_accuracy(predictions,labels)

            epoch_loss += loss.item() * batch_size
            epoch_acc += acc.item() * batch_size
        
    return epoch_loss / (len(train_dataset) - sp), epoch_acc / (len(train_dataset) - sp)
N_EPOCHS = 9

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train()
    valid_loss, valid_acc = evaluate()
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

torch.save(net.state_dict(), sys.argv[5])
