import pandas as pd
import sys
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from model import vgg16
import random
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

class MyDataset(Dataset):

	def __init__(self, file_path, transform=None):
		self.data = pd.read_csv(file_path)
		self.label_list = np.array(self.data.iloc[:, 0])
		temp = np.array(self.data.iloc[:, 1:])
		image_list = []
		for i in range(0, temp.shape[0]):
			a = np.fromstring(temp[i, 0], dtype = np.float32, sep = ' ').reshape(48, 48, 1)
			image_list.append(a)
		self.image_list = np.array(image_list)
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
	    # load image as ndarray type (Height * Width * Channels)
	    # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
	    # in this example, i don't use ToTensor() method of torchvision.transforms
	    # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
		
		image = self.image_list[index]
		label = self.label_list[index]
		if self.transform is not None:
		    image = self.transform(image)
		return image, label

train_dataset = MyDataset(sys.argv[1], transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.RandomHorizontalFlip(p = 0.3), 
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.8, 1), shear=15, resample=False, fillcolor=0),
	transforms.ToTensor()]))
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
batch_size = 64
sp = int(np.floor(dataset_size*0.8))
train_indices, val_indices = indices[:sp], indices[sp:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler)
valid_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = valid_sampler)

net = vgg16()
if use_cuda:
	net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 1e-4)
loss_list, acc_list = [], []
loss_list1, acc_list1 = [], []

def train(epoch):
	net.train()
	for i, (images, labels) in enumerate(train_loader):
		if use_cuda:
			images, labels = images.cuda(), labels.cuda()
		optimizer.zero_grad()

		output = net(images)

		loss = criterion(output, labels)

		if i % 10 == 0:
		    print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

		loss.backward()
		optimizer.step()
def valid():
	net.eval()
	total_correct = 0
	avg_loss = 0.0
	for i, (images, labels) in enumerate(valid_loader):
		if use_cuda:
			images, labels = images.cuda(), labels.cuda()
		output = net(images)
		avg_loss += criterion(output, labels).sum().item()
		pred = output.data.max(1)[1]
		total_correct += pred.eq(labels.data.view_as(pred)).sum().item()

	avg_loss /= (dataset_size - sp)
	acc = float(total_correct) / (dataset_size - sp)
	print('Valid Avg. Loss: %f, Accuracy: %f' % (avg_loss, acc))
	acc_list.append(acc)
	loss_list.append(avg_loss)
	total_correct = 0
	avg_loss = 0.0
	for i, (images, labels) in enumerate(valid_loader):
		if use_cuda:
			images, labels = images.cuda(), labels.cuda()
		output = net(images)
		avg_loss += criterion(output, labels).sum().item()
		pred = output.data.max(1)[1]
		total_correct += pred.eq(labels.data.view_as(pred)).sum().item()

	avg_loss /= (dataset_size - sp)
	acc = float(total_correct) / (dataset_size - sp)
	print('Valid Avg. Loss: %f, Accuracy: %f' % (avg_loss, acc))
	acc_list1.append(acc)
	loss_list1.append(avg_loss)
'''
def plotData(plt, x_data, y_data, y1_data, y_label):
	x = [p for p in x_data]
	y = [q for q in y_data]
	y1 = [r for r in y1_data]
	plt.title('Learning Curve')
	plt.xlabel('Epoch')
	plt.ylabel(y_label)
	plt.plot(x, y, '-.', label = 'valid')
	plt.plot(x, y1, '-.', label = 'train')
	plt.savefig(y_label)
	plt.close('all')
'''
def train_and_test(epoch):
	train(epoch)
	valid()
	

for e in range(1, 500):
	train_and_test(e)
'''
epoch_list = list(range(1, 500))
plotData(plt, epoch_list, acc_list, acc_list1, 'Training accuracy')
plotData(plt, epoch_list, loss_list, loss_list1, 'Training loss')
'''
torch.save(net.state_dict(), sys.argv[2])