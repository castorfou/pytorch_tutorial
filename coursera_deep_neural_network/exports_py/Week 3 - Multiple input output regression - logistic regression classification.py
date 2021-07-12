#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression  Prediction

# ## Class Linear

# In[4]:


import torch
from torch.nn import Linear
torch.manual_seed(1)
model = Linear(in_features=2, out_features=1)


# In[5]:


list(model.parameters())


# In[6]:


model.state_dict()


# In[7]:


X=torch.tensor([[1.0, 3.0]])
yhat=model(X)
yhat


# In[8]:


#predictions for multiple samples
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
yhat = model(X)
yhat


# ## Custom Modules

# In[9]:


import torch.nn as nn

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        out = self.linear(x)
        return out


# # Multiple Linear Regression Training

# ## Cost function and Gradient Descent for Multiple Linear Regression

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ## Train the model in Pytorch

# In[1]:


from torch import nn, optim
import torch

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        out = self.linear(x)
        return out


# In[6]:


from torch.utils.data import Dataset, DataLoader

class Data2D(Dataset):
    def __init__(self):
        self.x = torch.zeros(20,2)
        self.x[:, 0] = torch.arange(-1,1,0.1)
        self.x[:, 1] = torch.arange(-1,1,0.1)
        self.w = torch.tensor([ [1.0], [1.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w)+self.b
        self.y = self.f + 0.1*torch.randn((self.x.shape[0], 1))
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len


# In[7]:


data_set = Data2D()
criterion = nn.MSELoss()
trainloader = DataLoader(dataset=data_set, batch_size=2)
model = LR(input_size=2, output_size=1)
optimizer = optim.SGD(model.parameters(), lr=0.1)


# In[8]:


for epoch in range(100):
    for x, y in trainloader:
        yhat = model(x)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        


# # Multiple output linear regression

# ![image.png](attachment:image.png)

# With M linear functions with d inputs:
# ![image.png](attachment:image.png)

# ## Custom module

# In[3]:


import torch.nn as nn
import torch
class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        out = self.linear(x)
        return out


# In[4]:


torch.manual_seed(1)
model = LR(input_size=2, output_size=2)


# In[5]:


list(model.parameters())


# In[6]:


x=torch.tensor([[1.0, 2.0]])
yhat=model(x)
yhat


# In[7]:


#with 2 columns and 3 rows
X=torch.tensor([[1.0, 1.0], [1.0,2.0], [1.0, 3.0]])
Yhat = model(X)
Yhat


# In[ ]:




