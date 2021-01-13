#!/usr/bin/env python
# coding: utf-8

# # Installation

# ```bash
# conda create -n pytorch python=3.8
# conda activate pytorch
# conda install -c pytorch pytorch=1.7.1 torchvision
# conda install jupyter
# conda install -c conda-forge jupyter_contrib_nbextensions
# ```

# In[1]:


get_ipython().system('conda env list')


# In[3]:


# Install a conda package in the current Jupyter kernel
import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} matplotlib')


# # Test and validation

# In[1]:


import torch
x = torch.rand(5, 3)
print(x)


# In[1]:


import torch
torch.cuda.is_available()


# In[ ]:




