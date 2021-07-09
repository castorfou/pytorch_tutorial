#!/usr/bin/env python
# coding: utf-8

# # Tensors 1D

# ## The basics

# ### tensors, dtype, type()

# In[2]:


import torch
a=torch.tensor([7,4,3,2,6])


# In[3]:


a


# In[4]:


a[4]


# In[5]:


a.dtype


# In[6]:


a.type()


# ### tensors of float

# In[7]:


a=torch.tensor([0.0, 1.0, 2.1, 3.0, 4.0])


# In[8]:


a


# In[9]:


a.dtype


# In[10]:


a.type()


# ### force tensor type

# In[11]:


a=torch.tensor([0.0,1.0,2.0,3.0,4.0], dtype=torch.int32)


# In[12]:


a.dtype


# ### FloatTensor

# In[13]:


import torch
a=torch.FloatTensor([0,1,2,3,4])


# In[14]:


a


# ### Convert with .type()

# In[15]:


import torch
a=torch.tensor([0,1,2,3,4])
a=a.type(torch.FloatTensor)


# In[17]:


a.type()


# ### size(), ndimension()

# In[19]:


a=torch.Tensor([0,1,2,3,4])


# In[20]:


a.size()


# In[21]:


a.ndimension()


# ### tensor vs Tensor
# 
# https://stackoverflow.com/questions/51911749/what-is-the-difference-between-torch-tensor-and-torch-tensor

# torch.tensor infers the dtype automatically, while torch.Tensor returns a torch.FloatTensor. I would recommend to stick to torch.tensor, which also has arguments like dtype, if you would like to change the type.

# ### convert 1D to 2D using view

# In[22]:


a=torch.Tensor([0,1,2,3,4])
a_col = a.view(5,1)


# In[23]:


a_col


# In[24]:


b_col = a.view(-1,1)
b_col


# In[25]:


a_col.ndimension()


# ### create torch tensors from numpy arrays: from_numpy, numpy()

# In[26]:


import numpy as np
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])


# In[27]:


torch_tensor = torch.from_numpy(numpy_array)
back_to_numpy = torch_tensor.numpy()


# In[28]:


torch_tensor


# In[29]:


back_to_numpy


# ### pandas series to torch tensors

# In[32]:


import pandas as pd
pandas_series = pd.Series([0.1, 2, 0.3, 10.1])
pandas_series


# In[33]:


pandas_to_torch = torch.from_numpy(pandas_series.values)


# In[34]:


pandas_to_torch


# ### tensor to list: tolist()

# In[35]:


this_tensor = torch.tensor([0, 1, 2, 3])
torch_to_list = this_tensor.tolist()
torch_to_list


# ### item to return a number

# In[36]:


new_tensor = torch.tensor([5, 2, 6, 1])
new_tensor[0]


# In[37]:


new_tensor[0].item()


# ### indexing and slicing

# In[39]:


c=torch.tensor([20, 1, 2, 3, 4])
c


# In[40]:


c[0]=100
c


# In[41]:


c[4]=0
c


# In[42]:


#from 1 to 3
d=c[1:4]
d


# In[44]:


c[3:5]=torch.tensor([300.0, 4.0])
c


# ## Basic Operations

# ### vector addition

# In[46]:


u = torch.tensor([1.0, 0.0])
v = torch.tensor([0.0, 1.0])
z=u+v
z


# ### Vector multiplication with a scalar

# In[47]:


y = torch.tensor([1, 2])
z = 2*y
z


# ### hadamard product

# In[48]:


u = torch.tensor([1, 2])
v = torch.tensor([3, 4])
z = u*v
z


# ### dot product (produit scalaire)

# gives an indication about how similar these vectors are

# In[49]:


u = torch.tensor([1, 2])
v = torch.tensor([3, 1])
result = torch.dot(u, v)
result


# ### adding constant to a tensor

# it uses broadcasting

# In[50]:


u = torch.tensor([1, 2, 3, -1])
z = u + 1
z


# ## Functions

# ### universal functions: mean, max

# In[52]:


a = torch.tensor([1, -1, 1, -1.0])
a.mean()


# In[53]:


b = torch.tensor([1, -2, 3, 4, 5])
b.max()


# ### mathematical functions

# In[54]:


np.pi


# In[55]:


x= torch.tensor([0, np.pi/2, np.pi])
x


# In[57]:


y=torch.sin(x)
y


# ### to plot math function: linspace

# In[58]:


# evenly spaced numbers into a specified interval
torch.linspace(-2, 2, 5)


# In[60]:


torch.linspace(-2, 2, 9)


# In[61]:


x = torch.linspace(0, 2 * np.pi, 100)
y = torch.sin(x)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(x.numpy(), y.numpy())


# # Tensors 2D

# ## Tensor creation in 2D

# In[62]:


a = [ [11, 12, 13], [21, 22, 23], [31, 32, 33] ]
A = torch.tensor(a)
A


# In[63]:


A.ndimension()


# In[64]:


A.shape


# In[65]:


A.size()


# In[66]:


#number of elements
A.numel()


# ## Indexing and slicing in 2D

# In[68]:


a = [ [11, 12, 13], [21, 22, 23], [31, 32, 33] ]
A = torch.tensor(a)
A


# In[69]:


A[0, 0:2]


# In[70]:


A[1:3,2]


# ## Basic operations in 2D

# ### addition

# In[71]:


X = torch.tensor([[1,0], [0,1]])
Y = torch.tensor([[2,1], [1,2]])
Z = X+Y
Z


# ### hadamard product

# In[72]:


Z = X*Y
Z


# ### matrix multiplication

# In[73]:


A = torch.tensor([ [0, 1, 1], [1, 0, 1]])
B = torch.tensor([ [1, 1], [1, 1], [-1, 1]])
C = torch.mm(A, B)
C


# In[74]:


A.shape


# In[75]:


B.shape


# In[77]:


A[0,:].shape


# # Derivatives in PyTorch

# ## Derivatives

# $$y(x)=x^2$$

# In[86]:


import torch

x = torch.tensor(2., requires_grad=True)
y = x ** 2

y.backward()

x.grad


# $$z(x)=x^2+2x+1$$

# In[92]:


import torch

x = torch.tensor(2, requires_grad=True)
z = x**2 + 2*x + 1
z.backward()
x.grad


# ## Partial derivatives

# $$f(u, v)=uv+u^2$$

# $$\frac{\partial f(u,v)}{\partial u} = v+2u$$
# 
# $$\frac{\partial f(u,v)}{\partial v} = u$$

# In[87]:


import torch
u = torch.tensor(1., requires_grad=True)
v = torch.tensor(2., requires_grad=True)

f = u*v + u**2


# In[88]:


f.backward()
v.grad


# In[89]:


u.grad


# # Simple Dataset

# ## Build a Dataset Class and Object

# In[95]:


from torch.utils.data import Dataset

class toy_set(Dataset):
    def __init__(self, length=100, transform=None):
        self.x = 2*torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.len = length
        self.transform = transform
    def __getitem__(self, index):
        sample=self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        return self.len


# In[97]:


dataset = toy_set()
len(dataset)


# In[98]:


dataset[0]


# In[99]:


for i in range(3):
    x, y = dataset[i]
    print(f'{i} x:{x}, y:{y}')


# ## Build a Dataset Transform (e.g. normalize or standardize)

# In[102]:


class add_mult(object):
    def __init__(self, addx=1, muly=1):
        self.addx = addx
        self.muly = muly
    def __call__(self, sample):
        x=sample[0]
        y=sample[1]
        x=x+self.addx
        y=y*self.muly
        sample=x, y
        return sample
    


# In[104]:


# applying the transform directly to the dataset
dataset = toy_set()
a_m = add_mult()
x_, y_ = a_m(dataset[0])
x_, y_


# In[105]:


# automatically apply the transform
a_m = add_mult()
dataset_ = toy_set(transform=a_m)
dataset_[0]


# ## Compose Transforms

# In[108]:


class mult(object):
    def __init__(self, mul=100):
        self.mul = mul

    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mul
        y = y * self.mul
        sample = x, y
        return sample


# In[109]:


from torchvision import transforms

data_transform = transforms.Compose([add_mult(), mult()])


# In[110]:


#applying composed transform on data directly
x_, y_ = data_transform(dataset[0])
x_, y_


# In[111]:


#or directly in dataset
dataset_tr = toy_set(transform=data_transform)
dataset_tr[0]


# # Dataset

# ## Dataset Class for images

# In[115]:


from PIL import Image
import pandas as pd
import os
from matplotlib.pyplot import imshow
from torch.utils.data import Dataset, DataLoader


# Download MNIST fashion dataset (only first 100 images)

# In[155]:


get_ipython().system('wget -N https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/labs/Week1/data/img.tar.gz -P ./resources/data/')
get_ipython().system('tar -xf ./resources/data/img.tar.gz -C ./resources/data')
get_ipython().system('wget  https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/labs/Week1/data/index.csv -O index.csv')
get_ipython().system('mv index.csv ./resources/data/')


# In[156]:


directory = "./resources/data"
csv_file = "index.csv"
csv_path = os.path.join(directory, csv_file)

data_name = pd.read_csv(csv_path)
data_name.head()


# In[134]:


len(data_name)


# In[160]:


import random
index_image = random.randint(0,99)
image_path = os.path.join(directory, data_name.iloc[index_image]['image'])
print(f'Random image path {image_path}')
image = Image.open(image_path)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[index_image]['category'])
plt.show()


# In[166]:


class Dataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        data_dir_csv_file = os.path.join(self.data_dir, csv_file)
        self.data_name = pd.read_csv(data_dir_csv_file)
        self.len = self.data_name.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        img_name=os.path.join(self.data_dir, self.data_name.iloc[idx, 1])
        image = Image.open(img_name)
        y = self.data_name.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return image, y


# In[167]:


dataset = Dataset(csv_file=csv_file, data_dir=directory)
dataset[0]


# In[168]:


dataset[0][0]


# ## Torch Vision Transforms

# In[169]:


import torchvision.transforms as transforms
transforms.CenterCrop(20)
transforms.ToTensor()
croptensor_data_transform = transforms.Compose( [ transforms.CenterCrop(20), transforms.ToTensor() ] )
dataset = Dataset(csv_file=csv_file, data_dir=directory, transform=croptensor_data_transform)
dataset[0][0].shape


# ## Torch Vision Datasets

# In[171]:


import torchvision.datasets as dsets
dataset = dsets.MNIST(root='./data', train = False, download = True, transform = transforms.ToTensor())


# In[ ]:




