import sent2vec
import numpy as np
import json
import os

model = sent2vec.Sent2vecModel()
model.load_model(os.path.join('sent2vec', 'src', 'torontobooks_unigrams.bin')) # Can load different pretrained model , download from https://github.com/epfml/sent2vec
def caption2vec(path="data/training_label.json"):
   raw_labels = json.load(open(path))
   ret = []	
   for entry in raw_labels:
      entry['caption'] = model.embed_sentences(entry['caption'])
      ret.append([entry["id"],entry["caption"]])
   return ret	


