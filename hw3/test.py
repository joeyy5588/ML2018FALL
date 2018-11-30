import pandas as pd
import sys
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from model import vgg16
import csv
from collections import Counter


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
test_dataset = MyDataset(sys.argv[1], transform = transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size = 1)
net_array = []
for i in range(8):
	net = vgg16()
	if use_cuda:
		net = net.cuda()
	filename = ('vgg16_' + str(i+1) + '.pt')
	net.load_state_dict(torch.load(filename))#vgg16_1.pt
	net.eval()
	net_array.append(net)


def test():
	with open(sys.argv[2], 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['id', 'label'])
		for i, (images, labels) in enumerate(test_loader):
			if use_cuda:
				images = images.cuda()
			vote = []
			for j in range(8):
				output = net_array[j](images)
				pred = output.data.max(1)[1]
				vote.append(pred.item())
			cnt = Counter(vote)
			answer = cnt.most_common(1)[0][0]
			spamwriter.writerow([labels.data.item(), str(answer)])

test()