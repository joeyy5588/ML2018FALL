from modelFeat import ModelFeat
import os
import torch.optim as optim
import torch.nn as nn
import torch
from torch import save as save_model
from torch import load as load_model

def binary_accuracy(preds, y):
   """
   Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
   """
   #round predictions to the closest integer
   rounded_preds = torch.round(preds)
   correct = (rounded_preds == y).float() #convert into float for division 
   acc = correct.sum()/len(correct)
   return acc

class Model():
    def __init__(self,cuda = False,label = ''):
        
        self.modelF = ModelFeat()
        self.cuda = cuda
        if cuda:
            print ("USES CUDA")
            self.modelF = self.modelF.cuda()
        self.optimF = optim.Adam(self.modelF.parameters(), lr = 1e-4)
        self.lossF = nn.BCELoss()
        self.label = label
        self.valid_acc = []
        self.train_acc = []
    def train(self,loader,valid,epochs):
   
       for e in range(epochs):
            print('EPOCH {}/{}'.format(e+1, epochs))
            self.modelF.train()
            total_acc = 0.0
            total_loss = 0.0
            for i, (features, caption, gt) in enumerate(loader):
                if self.cuda:
                    features, caption, gt = features.cuda(), caption.cuda(), gt.cuda()
                self.optimF.zero_grad()
                predF = self.modelF(features, caption).squeeze(1)    
                lossF = self.lossF(predF, gt)
                #print(predF, gt)
                lossF.backward()
                self.optimF.step()
                total_acc += binary_accuracy(predF, gt) * features.shape[0]
                total_loss += lossF * features.shape[0]
            
            print('Train: acc = {0:.4f}, loss = {0:.4f}'.format(total_acc / (loader.length),total_loss / (loader.length)))
            self.train_acc.append(total_acc / loader.length)

            self.modelF.eval()
            total_acc = 0.0
            total_loss = 0.0
            for i, (features, caption, gt) in enumerate(valid):
                if self.cuda:
                    features, caption, gt = features.cuda(), caption.cuda(), gt.cuda()
                predF = self.modelF(features, caption).squeeze(1)
                lossF = self.lossF(predF, gt)
                total_acc += binary_accuracy(predF, gt) * features.shape[0]
                total_loss += lossF * features.shape[0]
            
            print('Valid: acc = {0:.4f}, loss = {0:.4f}'.format(total_acc / (valid.length),total_loss / (valid.length)))    
            self.valid_acc.append(total_acc / valid.length)
            valid_acc = total_acc / valid.length
            valid_acc = int(valid_acc * 100)
            
            if valid_acc >= 70:
                save_model(self.modelF.state_dict(), os.path.join('model', 'chooser_{}_{}_{}.pt'.format(self.label,e+1, valid_acc)))
                print('save model at valid_acc = {}'.format(valid_acc))
    
    def load(self, model_name):
        self.modelF.load_state_dict(load_model(os.path.join('model', model_name)))
