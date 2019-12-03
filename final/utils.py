from torch.utils.data import Dataset
import json
import numpy as np
import os
import re
import random
from caption2vec import caption2vec
from parameters import *


# read video list
def parse_video_list(): 
   with open(os.path.join('data', 'training_label.json'), 'rb') as file:
      datas = json.load(file)
   video_name_list = [] 
   
   for (i, item) in enumerate(datas):
      video_name_list.append(item['id'] + '.npy')
   return video_name_list

# read features
def parse_features(video_name_list):
   dir_name = os.path.join('data', 'training_data', 'feat')
   features = []
   for file in video_name_list:
      x = np.load(os.path.join(dir_name, file))
      features.append(x)
   features = np.array(features, dtype = np.float32)
   print(features.shape)
   return features
   
class dataset(Dataset):
   def __init__(self):
      self.captions = caption2vec()
      video_name_list = parse_video_list()
      self.features = parse_features(video_name_list)
   def __getitem__(self, i):
      idx_right = random.randint(0, len(self.captions[i]) - 1)
      caption_right = self.captions[i][1][idx_right]
      class_idx = i
      while class_idx == i:
         class_idx = random.randint(0, self.__len__() - 1)
      idx_wrong = random.randint(0, len(self.captions[class_idx]) - 1)
      caption_wrong = self.captions[class_idx][1][idx_wrong]
      iscorrect = random.random() < 0.5
      if iscorrect:
         caption = caption_right
      else:
         caption = caption_wrong
      caption = np.array(caption, dtype = np.float32)
      idx_feature = random.randint(0, VIDEO_LEN - 1)
      return (self.features[i][idx_feature], caption, np.float32(iscorrect))
   def __len__(self):
      return len(self.features)
      
if __name__ == '__main__':
   a = dataset()
   # print(a.max_length)
   print(a[0][0].shape, a[0][1].shape)
   print(np.linalg.norm(a[1][1]))
