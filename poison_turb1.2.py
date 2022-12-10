import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
from torch.utils import data
from model import LES
from torch.autograd import Variable
#from penalty import DivergenceLoss2
from train import Dataset
import copy
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys 

stdoutOrigin=sys.stdout 
sys.stdout = open("log.txt", "w")


train_indices = list(range(0, 500))
valid_indices = list(range(500, 600))

train_direc = "/data/siddharth/sample_"
test_direc = "/data/siddharth/sample_"

time_range  = 6
output_length = 4
input_length = 26
learning_rate = 1e-4
dropout_rate = 0
kernel_size = 3
batch_size = 30
num_epochs = 10

parser = argparse.ArgumentParser()
parser.add_argument('--poison_train', help="To train model with poisoned data points", action='store_true')
parser.add_argument('--gen_poison', help="To generate poison data points", action='store_true')

args = parser.parse_args()

train_set = Dataset(train_indices, input_length + time_range - 1, 40, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length + time_range - 1, 40, 6, test_direc, True)

if args.gen_poison:

  print('Loading dataset')
  
  train_gen = data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
  valid_gen = data.DataLoader(dataset = valid_set, batch_size = batch_size, shuffle = False, num_workers = 4)
  
  print('Defining model')
  
  model = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
              dropout_rate = dropout_rate, time_range = time_range).to(device)
  
  optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas = (0.9, 0.999), weight_decay = 4e-4)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)
  loss_function = torch.nn.MSELoss()
  #regularizer = DivergenceLoss2(torch.nn.MSELoss())
  coef = 0
  
  
  def trainNN(model,xp=None,yp=None): 
    
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)
    torch.cuda.empty_cache()
    model.train()
    
    xp = xp[None,:]
    yp = yp[None,:]  
    
    max_batches = 3
    
    for epoch in range(num_epochs):
      for batch_num, batch_data in enumerate(train_gen):
        loss = 0
        xx = batch_data[0]
        yy = batch_data[1]
        if xp!=None and batch_num == max_batches-1:
          xx = torch.cat((xx,xp),0)
          yy = torch.cat((yy,yp),0)
        xx = xx.to(device)
        yy = yy.to(device)
    
        for y in yy.transpose(0,1):
            im = model(xx)
            xx = torch.cat([xx[:, 2:], im], 1)
      
            if coef != 0 :
                loss += loss_function(im, y) + coef*regularizer(im, y)
            else:
                loss += loss_function(im, y)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_num == max_batches-1:
          break
      if epoch%5==0 or epoch==num_epochs-1:
        print('Epoch {}/{}, Batches {}/{}, Loss: {:.6f}'.format(epoch+1, num_epochs, batch_num+1, max_batches, loss))
      
    weights = []
  
    for x in enumerate(model.parameters()):
      weights.append(x[1].clone().detach())  
      
    return weights
    
  
  
  
  def getDerivative(model,set,xp=None,yp=None):
    
    assert set=='train' or set=='valid'
    if set=='train':
      data_gen=train_gen
    else:
      data_gen=valid_gen
      
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    
    gradients = []
    
    for i,x in enumerate(model.parameters()):
      gradients.append(torch.zeros_like(x))
    
    max_batches = 3
      
    for batch_num, batch_data in enumerate(data_gen):
      loss = 0
      xx = batch_data[0]
      yy = batch_data[1]
      if xp!=None and yp!=None and batch_num == max_batches-1:
        xp = xp[None,:]
        yp = yp[None,:]
        xx = torch.cat((xx,xp),0)
        yy = torch.cat((yy,yp),0)
      xx = xx.to(device)
      yy = yy.to(device)
    
      for y in yy.transpose(0,1):
          im = model(xx)
          xx = torch.cat([xx[:, 2:], im], 1)
    
          if coef != 0 :
              loss += loss_function(im, y) + coef*regularizer(im, y)
          else:
              loss += loss_function(im, y)
              
      optimizer.zero_grad()    
      loss.backward() 
      if batch_num == max_batches-1:
          break
      
      for i,x in enumerate(model.parameters()):
        gradients[i]+=x.grad
  
  
    return loss, gradients
  
  
  
  def getDerivative2(data_list,model):
    
    data_gen = torch.utils.data.DataLoader(dataset = data_list,
                                               batch_size = 1,
                                               shuffle = False, num_workers = 0)
    loss = 0    
     
    torch.set_grad_enabled(True)   
                  
    for xx, yy in data_gen:
    
      xx = xx.to(device).requires_grad_()
      params = [{'params': xx, 'weight_decay': 0, 'lr':0}]
      optimizer1 = torch.optim.SGD(params) 
      yy = yy.to(device)
      xx1 = xx
      
      for y in yy.transpose(0,1):
      
          im = model(xx1)
          xx1 = torch.cat([xx1[:, 2:], im], 1)       
          if coef != 0 :
              loss += loss_function(im, y) + coef*regularizer(im, y)
          else:
              loss += loss_function(im, y)
          
      
      
      optimizer1.zero_grad()
      loss.backward()
      return xx.grad[0]
      
  
  def reverseMLP2(xp,yp,model):
    
    ww = trainNN(model,xp,yp)
    
    cost, dw = getDerivative(model,'valid')
    epsilon = 1e-8
  
    dxp = torch.zeros_like(xp).cuda()
  
    for epoch in range(num_epochs):
      
      _, dww = getDerivative(model,'train',xp,yp)
      ww = [ww[i] + learning_rate * dww[i] for i in range(len(ww))]
      with torch.no_grad():
        for i,p in enumerate(model.parameters()):
          p.copy_(ww[i])
  
      wwm = copy.deepcopy(ww)
      wwm = [wwm[i] + 0.5*epsilon*dw[i] for i in range(len(wwm))]
      #model2 = model
      model2 = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
              dropout_rate = dropout_rate, time_range = time_range).to(device)
      model2.load_state_dict(model.state_dict())
      with torch.no_grad():
        for i,p in enumerate(model2.parameters()):
          p.copy_(wwm[i])
      _, dw2 = getDerivative(model2,'train',xp,yp)
      dw2x = getDerivative2([(xp,yp)],model2)
      
      
      wwm = [wwm[i] - epsilon*dw[i] for i in range(len(wwm))]
      #model1 = model
      model1 = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
              dropout_rate = dropout_rate, time_range = time_range).to(device)
      model1.load_state_dict(model.state_dict())
      with torch.no_grad():
        for i,p in enumerate(model1.parameters()):
          p.copy_(wwm[i])  
      dw1x = getDerivative2([(xp,yp)],model1)
      _, dw1 = getDerivative(model1,'train',xp,yp)
      
      
      ddxp = (dw2x - dw1x)/epsilon
      ddw = ([(a - b)/epsilon for a, b in zip(dw2, dw1)])
      #pdb.set_trace()
      dxp = dxp - alpha*ddxp
      dw = [dw[i] - alpha*ddw[i] for i in range(len(dw))] 
      if epoch%5==0 or epoch==num_epochs-1:
        print('Reverse Epoch {}/{}'.format(epoch+1, num_epochs))
      
      
    return cost, dxp
  
  poison = []
  print('Performing the Bi-level optimization')
  
  npois = 50
  
  pois_itr = 0
  while pois_itr<npois:
  
    c = np.random.randint(len(train_set))
    xp = train_set[c][0]
    yp = train_set[c][1]
    yp = torch.tensor(np.random.normal(0, 1, size=yp.shape)).float()
    #yp = torch.tensor(np.zeros(yp.shape)).float()
  
    num_iter = 3
    
    alpha = 1e-4
    dr = 1e-1
    
    model_LES = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
              dropout_rate = dropout_rate, time_range = time_range).to(device)
  
    for i in range(num_iter): 
      
      cost, dxp = reverseMLP2(xp,yp,model_LES)
      dxp = F.normalize(dxp)
      xp = xp + alpha*dxp.cpu()
      if xp.isnan().any():
        break
      xp[xp>1]=1
      xp[xp<0]=0
      alpha = alpha * dr
      print('Poisoning point: {}, Iteration: {}, Cost: {:.4f}'.format(pois_itr+1,i,cost))
    
    if xp.isnan().any():
        print('Poisoning point discarded due to Nan values')
        continue
    poison = poison + [(xp,yp)]
    torch.save(poison, 'pois_points.pt')
    pois_itr += 1
  
  
  
if args.poison_train:  
  
  print('Loading dataset')
  
  train_batch_size = 20
  valid_batch_size = 30
  
  train_gen = data.DataLoader(dataset = train_set, batch_size = train_batch_size, shuffle = True, num_workers = 2)
  valid_gen = data.DataLoader(dataset = valid_set, batch_size = valid_batch_size, shuffle = False, num_workers = 2)
  
  
  loss_function = torch.nn.MSELoss()
  #regularizer = DivergenceLoss2(torch.nn.MSELoss())
  coef = 0
  
  def train_poison(model,pois_list): 
    
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)
    torch.cuda.empty_cache()
    model.train()
    
    max_batches = 18
    
    for epoch in range(num_epochs):      
      for batch_num, batch_data in enumerate(train_gen):
        loss=0
        xx = batch_data[0]
        yy = batch_data[1]
        
        if batch_num<len(pois_list):
          pnt = pois_list[batch_num]
          xp = pnt[0][None,:]
          yp = pnt[1][None,:]
          xx = torch.cat((xx,xp),0)
          yy = torch.cat((yy,yp),0)
        xx = xx.to(device)
        yy = yy.to(device)
        
        for y in yy.transpose(0,1):
            im = model(xx)
            xx = torch.cat([xx[:, 2:], im], 1)
      
            if coef != 0 :
                loss += loss_function(im, y) + coef*regularizer(im, y)
            else:
                loss += loss_function(im, y)
        
        #pdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_num == max_batches-1:
          break
          
      if epoch%5==0 or epoch==num_epochs-1:
        print('Epoch {}/{}, Loss: {:.6f}'.format(epoch+1, num_epochs, loss))
    
  
  
  Valid_loss_pois = []
  poison = torch.load('pois_points.pt')
  
  
  # Specify the number of poison points to include in training
  num_poison_list = [0,1,2,3,4,5]
  
  for i in range(len(num_poison_list)):
    model_LES = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
              dropout_rate = dropout_rate, time_range = time_range).to(device)
  
    optimizer = torch.optim.SGD(model_LES.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)
    
    model_LES.train()  
  
    print('Training on training dataset + {} poison points'.format(num_poison_list[i]))
    train_poison(model_LES,poison[:num_poison_list[i]])
    
    loss = 0
    
    model_LES.eval()        
                                        
    for xx, yy in valid_gen:
      
      xx = xx.to(device)
      yy = yy.to(device)
    
      for y in yy.transpose(0,1):
          im = model_LES(xx)
          xx = torch.cat([xx[:, 2:], im], 1)  
          loss += loss_function(im, y)
    
    loss = loss/((len(train_indices)/train_batch_size))
    
    print('Validation Loss = {}'.format(loss.detach()))  
    Valid_loss_pois.append(loss.detach().cpu())
    torch.save(Valid_loss_pois, 'Valid_loss_pois.pt')
  
  torch.save(Valid_loss_pois, 'Valid_loss_pois.pt')

sys.stdout.close()
sys.stdout=stdoutOrigin


