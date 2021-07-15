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

# # Neural networks in One Dimension

# ## using PyTorch with nn.Module

# In[1]:


import torch
import torch.nn as nn
from torch import sigmoid


# In[2]:


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x=sigmoid(self.linear1(x))
        x=sigmoid(self.linear2(x))
        return x
    


# In[3]:


model = Net(1, 2, 1)
x = torch.tensor([0.0])
yhat = model(x)
yhat


# In[12]:


# multiple samples
x = torch.tensor([[0.0], [2.0], [3.0]])
yhat = model(x)
yhat


# In[13]:


# to get a discrete value we apply a threshold
yhat = yhat < 0.59
yhat


# In[14]:


model.state_dict()


# ## using PyTorch with nn.Sequential

# In[15]:


model = nn.Sequential(nn.Linear(1, 2), nn.Sigmoid(), nn.Linear(2, 1), nn.Sigmoid())


# ## Train the model

# ### we create the data

# In[16]:


X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0]>-4) & (X[:, 0] <4)] = 1.0


# ### we create a training function

# In[41]:


from tqdm import tqdm

def train(Y, X, model, optimizer, criterion, epochs=1000):
    cost = []
    total = 0
    for epoch in tqdm(range(epochs)):
        total = 0
        for x, y in zip(X, Y):
            yhat = model(x)
            loss = criterion(yhat, y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total+=loss.item()
        cost.append(total)
    return cost
            


# ### training process

# In[42]:


#loss
criterion = nn.BCELoss()

#data
X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0]>-4) & (X[:, 0] <4)] = 1.0

#model
model = Net(1, 2, 1)

#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#train the model
cost = train(Y, X, model, optimizer, criterion, epochs=1000)


# # Neural Networks More Hidden Neurons

# ## in PyTorch

# In[43]:


import torch
import torch.nn as nn
from torch import sigmoid
from torch.utils.data import Dataset, DataLoader


# class to get our dataset

# In[50]:


class Data(Dataset):
    def __init__(self):
        self.x = torch.linspace(-20, 20, 100).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x[:, 0]>-10) & (self.x[:, 0]<-5)] = 1
        self.y[(self.x[:, 0]>5) & (self.x[:, 0]<10)] = 1
        self.y = self.y.view(-1, 1)
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len


# class for creating our model

# In[51]:


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x=sigmoid(self.linear1(x))
        x=sigmoid(self.linear2(x))
        return x


# and the function to train our model

# In[61]:


# The function for plotting the model
def PlotStuff(X, Y, model):
    
    plt.plot(X.numpy(), model(X).detach().numpy())
    plt.plot(X.numpy(), Y.numpy(), 'r')
    plt.xlabel('x')


# In[65]:


def train(data_set, model, criterion, train_loader, optimizer, epochs=5):
    cost = []
    total=0
    for epoch in tqdm(range(epochs)):
        total=0
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            total+=loss.item() 
            PlotStuff(data_set.x, data_set.y, model)
        cost.append(total)
    return cost


# process for training is identical to logistic regression

# In[63]:


criterion = nn.BCELoss()
data_set = Data()
train_loader = DataLoader(dataset=data_set, batch_size=100)
model = Net(1, 6, 1)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
train(data_set, model, criterion, train_loader, optimizer, epochs=1000)


# ## using sequential

# In[64]:


model = nn.Sequential(
    nn.Linear(1, 7),
    nn.Sigmoid(),
    nn.Linear(7, 1),
    nn.Sigmoid()
)


# # Neural Networks with Multiple Dimensional Input

# ## in PyTorch

# In[97]:


import torch
import torch.nn as nn
from torch import sigmoid
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


# we create a dataset class

# In[98]:


class XOR_Data(Dataset):
    def __init__(self, N_s=100):
        self.x = torch.zeros((N_s, 2))
        self.y = torch.zeros((N_s, 1))
        for i in range(N_s // 4):
            self.x[i, :] = torch.Tensor([0.0, 0.0])
            self.y[i, 0] = torch.Tensor([0.0])
            self.x[i + N_s // 4, :] = torch.Tensor([0.0, 1.0])
            self.y[i + N_s // 4, 0] = torch.Tensor([1.0])
            self.x[i + N_s // 2, :] = torch.Tensor([1.0, 0.0])
            self.y[i + N_s // 2, 0] = torch.Tensor([1.0])
            self.x[i + 3 * N_s // 4, :] = torch.Tensor([1.0, 1.0])
            self.y[i + 3 * N_s // 4, 0] = torch.Tensor([0.0])
            self.x = self.x + 0.01 * torch.randn((N_s, 2))
        self.len = N_s
            
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len      
    # Plot the data
    def plot_stuff(self):
        plt.plot(self.x[self.y[:, 0] == 0, 0].numpy(), self.x[self.y[:, 0] == 0, 1].numpy(), 'o', label="y=0")
        plt.plot(self.x[self.y[:, 0] == 1, 0].numpy(), self.x[self.y[:, 0] == 1, 1].numpy(), 'ro', label="y=1")
        plt.legend()


# In[99]:


data = XOR_Data()
data.plot_stuff()


# we create a class for creating our model

# In[100]:


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x=sigmoid(self.linear1(x))
        x=sigmoid(self.linear2(x))
        return x


# we create a function to train our model

# In[101]:


# Calculate the accuracy

def accuracy(model, data_set):
    return np.mean(data_set.y.view(-1).numpy() == (model(data_set.x)[:, 0] > 0.5).numpy())


# In[102]:


def train(data_set, model, criterion, train_loader, optimizer, epochs=5):
    COST = []
    ACC = []
    for epoch in tqdm(range(epochs)):
        total=0
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #cumulative loss 
            total+=loss.item()
        ACC.append(accuracy(model, data_set))
        COST.append(total)
        
    return COST


# process for training is identical to logistic regression

# In[103]:


criterion = nn.BCELoss()
data_set = XOR_Data()
train_loader = DataLoader(dataset=data_set, batch_size=1)
model = Net(2, 4, 1)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
train(data_set, model, criterion, train_loader, optimizer, epochs=500)


# ## overfitting and underfitting

# ![image-2.png](attachment:image-2.png)

# ![image.png](attachment:image.png)

# Solution
# * use validation data to determine optimum number of neurons
# * get more data
# * regularization: for example dropout

# # Multi-Class Neural Networks

# ## using nn.Module

# In[104]:


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x=sigmoid(self.linear1(x))
        x=(self.linear2(x))
        return x


# ## using nn.Sequential

# In[107]:


input_dim = 2
hidden_dim = 6
output_dim = 3
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.Sigmoid(),
    nn.Linear(hidden_dim, output_dim)
)


# ## training

# we create a validation and training dataset

# In[109]:


import torchvision.datasets as dsets
import torchvision.transforms as transforms
train_dataset = dsets.MNIST(root='./data', train = True, download = True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train = False, download = True, transform=transforms.ToTensor())


# we create a validation and training loader

# In[114]:


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=2000)


# In[115]:


criterion = nn.CrossEntropyLoss()


# we create the training function

# In[116]:


from tqdm import tqdm
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    useful_stuff = {'training_loss': [],'validation_accuracy': []}  
    for epoch in tqdm(range(epochs)):
        for i, (x, y) in enumerate(train_loader): 
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
             #loss for every iteration
            useful_stuff['training_loss'].append(loss.data.item())
        correct = 0
        for x, y in validation_loader:
            #validation 
            z = model(x.view(-1, 28 * 28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()
        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    return useful_stuff


# We instantiate and Train the model

# In[119]:


input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

model = Net(input_dim, hidden_dim, output_dim)

training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=30)


# Plot improper classified items

# In[ ]:



count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _,yhat = torch.max(z, 1)
    if yhat != y:
        show_data(x)
        count += 1
    if count >= 5:
        break


# # Backpropagation

# Following rule chain for gradient calculation, it happens that gradient result are getting closer and closer to 0. (i.e. vanishing gradient) therefore we cannot improve model parameters.
# 
# One way to deal with that is to change activation function.
# 

# # Activation functions

# ## sigmoid, tanh, relu activation functions

# ## sigmoid, tanh, relu in PyTorch

# In[120]:


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x=sigmoid(self.linear1(x))
        x=(self.linear2(x))
        return x


# In[121]:


class Net_tanh(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x=torch.tanh(self.linear1(x))
        x=(self.linear2(x))
        return x


# In[122]:


class Net_relu(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x=torch.relu(self.linear1(x))
        x=(self.linear2(x))
        return x


# In[123]:


model_tanh = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, output_dim)
)

model_relu = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)


# ![image.png](attachment:image.png)

# In[ ]:




