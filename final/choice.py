from model import Model
from parameters import *
import numpy as np 
import os 
import csv
import sys
import torch
import re
import sent2vec
from torch.utils.data import Dataset,DataLoader
from torch.nn.functional import mse_loss
from utils import dataset

use_cuda = torch.cuda.is_available()
model = Model(use_cuda)

s2v_model = sent2vec.Sent2vecModel()
s2v_model.load_model(os.path.join('sent2vec', 'src', 'torontobooks_unigrams.bin'))

PATH_TO_DATA = sys.argv[3]

def parse_features(options):
    dir_name = os.path.join(PATH_TO_DATA, 'testing_data', 'feat')
    features = []
    for video in options:
        x = np.load(os.path.join(dir_name,video[0]+".npy"))
        features.append(x)
    features = np.array(features, dtype = np.float32)
    print(features.shape)
    return features

def parse_optionsNfeatures():
    file = open(os.path.join(PATH_TO_DATA, 'testing_options.csv'), encoding='UTF-8-sig')
    option_csv = csv.reader(file)
    options = (list(option_csv))
    options_list = []

    for video in options:
        video_option = s2v_model.embed_sentences(video[1:6]) 
        options_list.append(video_option)
        
    return parse_features(options), options_list

class test_dataset(Dataset):
    def __init__(self):
       self.features, self.captions = parse_optionsNfeatures()
    def __getitem__(self, i):
        return (self.features[i], self.captions[i])
    def __len__(self):
        return len(self.features)

def predict(label):
    model.load(label)
    test_data = test_dataset()
    loader = DataLoader(test_data,batch_size = 500, shuffle = False)
    modelF = model.modelF
    modelF.eval()
    results = np.zeros([500,5])

    for i, (feature, captions) in enumerate(loader):
        if use_cuda:
            feature, captions = feature.cuda(), captions.cuda()
        for option_no in range(5):
            print ("TESTING OPTIONS %d"%option_no)
            for time in range(VIDEO_LEN):
                pred = modelF(feature.select(1,time),captions[:,option_no,:])
                results[:,option_no] += pred.squeeze(1).data.cpu().numpy()
    ans = np.argmax(results,1)
    ans_oh = np.zeros([500,5])
    for i,j in enumerate(ans):
        ans_oh[i,j] = 1
    return ans_oh,results
