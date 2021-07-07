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


# ## indexing and slicing

# In[ ]:




