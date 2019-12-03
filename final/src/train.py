from utils import dataset
from torch.utils.data import DataLoader
import torch
import sys
from model import Model
from parameters import *
label = sys.argv[1]
# detect gpu
use_cuda = torch.cuda.is_available()

data = dataset()
train_size,valid_size = int(0.8*len(data)) ,int(0.2*len(data))
train_data, valid_data = torch.utils.data.random_split(data, [train_size, valid_size])

train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = BATCH_SIZE, shuffle = False)

train_loader.length = len(train_data)
valid_loader.length = len(valid_data)
model = Model(use_cuda, label)
#model.load("chooser_1223_75.pt")

model.train(train_loader,valid_loader,400)
