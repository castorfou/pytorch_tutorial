#!/usr/bin/env python
# coding: utf-8

# https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/

# # Quick Fix: How To Install Packages from the Jupyter Notebook

# ![image.png](attachment:image.png)

# In[1]:


# Install a conda package in the current Jupyter kernel
import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} matplotlib')


# # How your operating system locates executables

# In[ ]:




