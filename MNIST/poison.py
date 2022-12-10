import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import tqdm



print('Loading dataset')

train_data = dsets.MNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])

test_data = dsets.MNIST(root = './data', train = False,
                       transform = transforms.ToTensor())

train_list = []
for i in range(len(train_data)):
  train_list.append(train_data[i])

val_list = []
for i in range(len(val_data)):
  val_list.append(val_data[i])

test_list = []
for i in range(len(test_data)):
  test_list.append(test_data[i])


print('Defining model')

input_size = 784 # img_size = (28,28) ---> 28*28=784 in total
hidden_size = 500 # number of nodes at hidden layer
num_classes = 10 # number of output classes discrete range [0,9]
num_epochs = 10 # number of times which the entire dataset is passed throughout the model
batch_size = 100 # the size of input data took for one iteration
lr = 1e-3 # size of step 


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
      super(Net,self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
      out = self.fc1(x)
      out = self.relu(out)
      out = self.fc2(out)
      return out

net = Net(input_size, hidden_size, num_classes)

if torch.cuda.is_available():
    net.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( net.parameters(), lr=lr)


def trainNN(tr_list,disp=True): 
  train_gen = torch.utils.data.DataLoader(dataset = tr_list,
                                             batch_size = 100,
                                             shuffle = False)
  
  for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_gen):
      images = Variable(images.view(-1,28*28)).cuda()
      labels = Variable(labels.view(-1)).cuda()
      
      optimizer.zero_grad()
      outputs = net(images)
      loss = loss_function(outputs, labels)
      loss.backward()
      optimizer.step() 
        
    if disp==True and (epoch%5==0 or epoch==num_epochs-1):
      print('Epoch {}/{}, Loss: {:.6f}'.format(epoch+1, num_epochs, loss))
  
  
  weights = []

  for x in enumerate(net.parameters()):
    weights.append(x[1])
  
  return weights


# Training and validation results on clean data
print('Training and validation results on clean data')

trainNN(train_list)
val_gen = torch.utils.data.DataLoader(dataset = val_list[:100],
                                             batch_size = 100,
                                             shuffle = False)
  
for i ,(images,labels) in enumerate(val_gen):
  images = Variable(images.view(-1,28*28)).cuda().requires_grad_()
  labels = Variable(labels.view(-1)).cuda()
  outputs = net(images)
  loss = loss_function(outputs, labels)
  
print('Validation Loss on first 100 validation points = {:.4f}'.format(loss))

# Re-initialising the model
net = Net(input_size, hidden_size, num_classes)

if torch.cuda.is_available():
    net.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( net.parameters(), lr=lr)

def getDerivative(data_list,model):
  
  data_gen = torch.utils.data.DataLoader(dataset = data_list,
                                             batch_size = len(data_list),
                                             shuffle = False)
  
  for i ,(images,labels) in enumerate(data_gen):
    images = Variable(images.view(-1,28*28)).cuda()
    labels = Variable(labels.view(-1)).cuda()
    outputs = model(images)
    loss = loss_function(outputs, labels)
    loss.backward()
    
    gradients = []

    for x in enumerate(model.parameters()):
      gradients.append(x[1].grad)
    
    return loss, gradients

def getDerivative2(data_list,model):
  
  data_gen = torch.utils.data.DataLoader(dataset = data_list,
                                             batch_size = len(data_list),
                                             shuffle = False)
  
  for i ,(images,labels) in enumerate(data_gen):
    images = Variable(images.view(-1,28*28)).cuda().requires_grad_()
    labels = Variable(labels.view(-1)).cuda()
    outputs = model(images)
    loss = loss_function(outputs, labels)
    loss.backward()

    
  return images.grad.view(1,28,28)

def reverseMLP2(xp,yp):
  nval = len(val_list)
  tr_list = train_list + [(xp,yp)]
  ww = trainNN(tr_list)
  
  cost, dw = getDerivative(val_list,net)

  epsilon = 1e-8

  dxp = torch.zeros_like(xp).cuda()

  for i in range(num_epochs):
    
    _, dww = getDerivative(tr_list,net)
    ww = [ww[i] + alpha*dww[i] for i in range(len(ww))]
    with torch.no_grad():
      for i,p in enumerate(net.parameters()):
        p.copy_(ww[i])

    wwm = ww
    wwm = [wwm[i] + 0.5*epsilon*dw[i] for i in range(len(wwm))]
    net2 = net
    with torch.no_grad():
      for i,p in enumerate(net2.parameters()):
        p.copy_(wwm[i])
    dw2x = getDerivative2([tr_list[-1]],net2)
    _, dw2 = getDerivative(tr_list,net2)
    
    wwm = [wwm[i] - epsilon*dw[i] for i in range(len(wwm))]
    net1 = net
    with torch.no_grad():
      for i,p in enumerate(net1.parameters()):
        p.copy_(wwm[i])   
    dw1x = getDerivative2([tr_list[-1]],net1)
    _, dw1 = getDerivative(tr_list,net1)
    
    ddxp = (dw2x - dw1x)/epsilon
    ddw = ([(a - b)/epsilon for a, b in zip(dw2, dw1)])
    dxp = dxp - alpha*ddxp
    dw = [dw[i] - alpha*ddw[i] for i in range(len(dw))]  

  return cost, dxp

poison = []
print('Performing the Bi-level optimization')

npois = 1

for j in range(npois):

  net = Net(input_size, hidden_size, num_classes)

  if torch.cuda.is_available():
      net.cuda()

  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam( net.parameters(), lr=lr)
  alpha = 1e-4
  dr = 1e-1
  c = np.random.randint(len(val_list))
  xp = val_data[c][0]
  yp = val_data[c][1]
  yp = torch.tensor([np.random.randint(10),])

  num_iter = 3

  for i in range(num_iter):
    cost, dxp = reverseMLP2(xp,yp)
    dxp = F.normalize(dxp)
    
    xp = xp + alpha*dxp.cpu()
    xp[xp>1]=1
    xp[xp<0]=0
    print('Poisoning point: {}, Iteration: {}, Cost: {:.4f}'.format(j+1,i,cost))
    alpha = alpha * dr
  poison = poison + [(xp,yp)]
  torch.save(poison, 'pois_points.pt')
    


Valid_loss_pois = []
num_pois = [0,505,1020,1546,2083,2631]
poison = torch.load('pois_points.pt')
print(len(poison))

for i in num_pois:
  net = Net(input_size, hidden_size, num_classes)

  if torch.cuda.is_available():
      net.cuda()

  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam( net.parameters(), lr=lr)

  print('Training on train-set + {} poison points'.format(i))
  trainNN(train_list + poison[:i])
  val_gen = torch.utils.data.DataLoader(dataset = val_list,
                                              batch_size = len(val_list),
                                              shuffle = False)
    
  for i ,(images,labels) in enumerate(val_gen):
    images = Variable(images.view(-1,28*28)).cuda().requires_grad_()
    labels = Variable(labels.view(-1)).cuda()
    outputs = net(images)
    loss = loss_function(outputs, labels)
    
    Valid_loss_pois.append(loss)


Valid_loss_pois = [x.detach().cpu() for x in Valid_loss_pois]
torch.save(Valid_loss_pois, 'Valid_loss_pois.pt')

'''

import matplotlib.pyplot as plt
plt.plot(Valid_loss_pois)
plt.xlabel('Percentage of poisoning points in training set')
plt.ylabel('Validation Loss')
plt.savefig('Valid_loss.jpg')
plt.show()

'''
