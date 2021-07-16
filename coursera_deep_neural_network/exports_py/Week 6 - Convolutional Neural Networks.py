#!/usr/bin/env python
# coding: utf-8

# # Convolution

# In[1]:


import torch
import torch.nn as nn


# In[2]:


conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)


# In[3]:


image = torch.zeros(1,1,5,5)
image[0,0,:,2] = 1
image


# In[4]:


z=conv(image)
z


# In[10]:


conv.state_dict()


# ## size of activation map

# ### stride

# In[8]:


conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride = 2)


# ### zeros padding

# In[9]:


conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride = 2, padding = 1)


# # Activation Functions and Max Polling

# ## Activation function using nn.Module

# In[11]:


import torch
image = torch.zeros(1,1,5,5)
image[0,0,:,2] = 1
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
z=conv(image)
A = torch.relu(z)


# ## Activation function using nn.Sequential

# In[12]:


relu = nn.ReLU()
A = relu(z)


# ## Max pooling

# In[13]:


max = nn.MaxPool2d(2, stride=1)
max(image)


# In[14]:


torch.max_pool2d(image, stride=1, kernel_size=2)


# # Multiple Input and Output Channels

# # Convolutional Neural Network

# ## using nn.Module

# ## training

# # GPU in PyTorch

# In[16]:


torch.cuda.is_available()


# In[17]:


device = torch.device('cuda:0')


# In[18]:


torch.tensor([1,2,32,4]).to(device)


# In[20]:


class CNN(nn.Module):
    
    # Contructor
    def __init__(self, out_1=16, out_2=32):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, 10)
    
    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# In[21]:


model = CNN()
model.to(device)


# ## training

# In[ ]:


for epoch in range(num_epochs):
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()


# # TORCH-VISION MODELS

# ## pretrained

# In[23]:


import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0)

model = models.resnet18(pretrained=True)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

composed = transforms.Compose([transforms.Resize(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)])

train_dataset = Dataset(transform=composed, train = True)
validation_dataset = Dataset(transform=composed)


# In[24]:


for param in model.parameters():
    param.requires_grad=False
model.fc = nn.Linear(512, 7)


# In[25]:


train_loader = DataLoader(dataset=train_loader, batch_size=15)
validation_loader = DataLoader(dataset=validation_loader, batch_size=10)


# In[26]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr = 0.003)

N_EPOCHS = 20
loss_list = []
accuracy_list = []
correct = 0
n_test = len(validation_dataset)


# In[ ]:


for epoch in range(N_EPOCHS):
    loss_sublist = []
    for x, y in train_loader:
        model.train()
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss_sublist.append(loss.data.item())
        loss.backward()
        optimizer.step()
    loss_list.append(np.mean(loss_sublist))
    correct = 0
    for x_test, y_test in validation_loader:
        model.eval()
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
    accuracy = correct / n_test
    accuracy_list.append(accuracy)

