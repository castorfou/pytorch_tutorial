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


# In[ ]:




