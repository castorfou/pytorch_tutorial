#!/usr/bin/env python
# coding: utf-8

# # Softmax prediction

# Softmax is a combination of logistic regression and argmax:
# ![image.png](attachment:image.png)

# # Softmax function

# ## Custom module using nn.Module

# In[1]:


import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, in_size, out_size):
        super(Softmax, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    def forward(self, x):
        out = self.linear(x)
        return out


# In[3]:


import torch
torch.manual_seed(1)
# 2 dimensions input samples and 3 output classes
model = Softmax(2,3)


# In[6]:


x = torch.tensor([[1.0, 2.0]])
z = model(x)
z


# In[7]:


_, yhat = z.max(1)
yhat


# In[12]:


X=torch.tensor([[1.0, 1.0],[1.0, 2.0],[1.0, -3.0]])
z = model(X)
z


# In[13]:


_, yhat = z.max(1)
yhat


# # Softmax PyTorch

# ## Load Data

# In[14]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets


# In[16]:


train_dataset = dsets.MNIST(root='./data', train = True, download = True, transform=transforms.ToTensor())

validation_dataset = dsets.MNIST(root='./data', train = False, download = True, transform=transforms.ToTensor())


# In[17]:


train_dataset[0]


# ## Create Model

# In[18]:


import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, in_size, out_size):
        super(Softmax, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    def forward(self, x):
        out = self.linear(x)
        return out


# In[19]:


input_dim = 28 * 28
output_dim = 10
model = Softmax(input_dim, output_dim)


# In PyTorch, when the loss criteria is specified as cross entropy loss, PyTorch will automatically perform Softmax classification based upon its inbuilt functionality.
# Another note, the input for the loss criterion here needs to be a long tensor with dimension of n, instead of n by 1 which we had used previously for linear regression. 

# In[20]:


criterion = nn.CrossEntropyLoss()


# In[21]:


import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.01)

n_epochs = 100
accuracy_list = []


# In[22]:


train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)


# # Train Model

# In[26]:


from tqdm import tqdm

for epoch in tqdm(range(n_epochs)):
    for x, y in train_loader:
        optimizer.zero_grad()
        z = model(x.view(-1, 28 * 28))
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
    correct = 0
    for x_test, y_test in validation_loader:
        z = model(x_test.view(-1, 28 * 28))
        _, yhat = torch.max(z.data, 1)
        correct = correct+(yhat == y_test).sum().item()
    accuracy = correct / y.shape[0]
    accuracy_list.append(accuracy)


# In[27]:


accuracy_list


# ## View Results

# ![image.png](attachment:image.png)

# In[ ]:




