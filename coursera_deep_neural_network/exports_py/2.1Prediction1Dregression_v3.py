#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/Template/module%201/images/IDSNlogo.png" width="300" alt="cognitiveclass.ai logo"  />
# </center>
# 

# <h1>Linear Regression 1D: Prediction</h1>
# 

# <h2>Objective</h2><ul><li> How to make the prediction for multiple inputs.</li><li> How to use linear class to build more complex models.</li><li> How to build a custom module.</li></ul> 
# 

# <h2>Table of Contents</h2>
# <p>In this lab, we will  review how to make a prediction in several different ways by using PyTorch.</h2>
# <ul>
#     <li><a href="#Prediction">Prediction</a></li>
#     <li><a href="#Linear">Class Linear</a></li>
#     <li><a href="#Cust">Build Custom Modules</a></li>
# </ul>
# <p>Estimated Time Needed: <strong>15 min</strong></p>
# 
# <hr>
# 

# <h2>Preparation</h2>
# 

# The following are the libraries we are going to use for this lab.
# 

# In[1]:


# These are the libraries will be used for this lab.

import torch


# <!--Empty Space for separating topics-->
# 

# <h2 id="Prediction">Prediction</h2>
# 

# Let us create the following expressions:
# 

# $b=-1,w=2$
# 
# $\hat{y}=-1+2x$
# 

# First, define the parameters:
# 

# In[2]:


# Define w = 2 and b = -1 for y = wx + b

w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)


# Then, define the function <code>forward(x, w, b)</code> makes the prediction: 
# 

# In[3]:


# Function forward(x) for prediction

def forward(x):
    yhat = w * x + b
    return yhat


# Let's make the following prediction at <i>x = 1</i>
# 

# $\hat{y}=-1+2x$
# 
# $\hat{y}=-1+2(1)$
# 

# In[4]:


# Predict y = 2x - 1 at x = 1

x = torch.tensor([[1.0]])
yhat = forward(x)
print("The prediction: ", yhat)


# <!--Empty Space for separating topics-->
# 

# Now, let us try to make the prediction for multiple inputs:
# 

# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter2/2.1.2.png" width="500" alt="Linear Regression Multiple Input Samples" />
# 

# Let us construct the <code>x</code> tensor first. Check the shape of <code>x</code>.
# 

# In[5]:


# Create x Tensor and check the shape of x tensor

x = torch.tensor([[1.0], [2.0]])
print("The shape of x: ", x.shape)


# Now make the prediction: 
# 

# In[6]:


# Make the prediction of y = 2x - 1 at x = [1, 2]

yhat = forward(x)
print("The prediction: ", yhat)


# The result is the same as what it is in the image above.
# 

# <!--Empty Space for separating topics-->
# 

# <h3>Practice</h3>
# 

# Make a prediction of the following <code>x</code> tensor using the <code>w</code> and <code>b</code> from above.
# 

# In[7]:


# Practice: Make a prediction of y = 2x - 1 at x = [[1.0], [2.0], [3.0]]

x = torch.tensor([[1.0], [2.0], [3.0]])
forward(x)


# Double-click <b>here</b> for the solution.
# 
# <!-- Your answer is below:
# yhat = forward(x)
# print("The prediction: ", yhat)
# -->
# 

# <!--Empty Space for separating topics-->
# 

# <h2 id="Linear">Class Linear</h2>
# 

# The linear class can be used to make a prediction. We can also use the linear class to build more complex models. Let's import the module:
# 

# In[8]:


# Import Class Linear

from torch.nn import Linear


# Set the random seed because the parameters are randomly initialized:
# 

# In[9]:


# Set random seed

torch.manual_seed(1)


# <!--Empty Space for separating topics-->
# 

# Let us create the linear object by using the constructor. The parameters are randomly created. Let us print out to see what <i>w</i> and <i>b</i>. The parameters of an <code>torch.nn.Module</code> model are contained in the model’s parameters accessed with <code>lr.parameters()</code>:
# 

# In[10]:


# Create Linear Regression Model, and print out the parameters

lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))


# This is equivalent to the following expression:  
# 

# $b=-0.44, w=0.5153$
# 
# $\hat{y}=-0.44+0.5153x$
# 

# A method  <code>state_dict()</code> Returns a Python dictionary object corresponding to the layers of each parameter  tensor. 
# 

# In[11]:


print("Python dictionary: ",lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())


# The keys correspond to the name of the attributes and the values correspond to the parameter value.
# 

# In[12]:


print("weight:",lr.weight)
print("bias:",lr.bias)


# Now let us make a single prediction at <i>x = [[1.0]]</i>.
# 

# In[13]:


# Make the prediction at x = [[1.0]]

x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)


# <!--Empty Space for separating topics-->
# 

# Similarly, you can make multiple predictions:
# 

# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter2/2.1.2vector_function.png" width="500" alt="Linear Class Sample with Multiple Inputs" />
# 

# Use model <code>lr(x)</code> to predict the result.
# 

# In[14]:


# Create the prediction using linear model

x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)


# <!--Empty Space for separating topics-->
# 

# <h3>Practice</h3>
# 

# Make a prediction of the following <code>x</code> tensor using the linear regression model <code>lr</code>.
# 

# In[15]:


# Practice: Use the linear regression model object lr to make the prediction.

x = torch.tensor([[1.0],[2.0],[3.0]])
lr(x)


# Double-click <b>here</b> for the solution.
# 
# <!-- Your answer is below:
# x=torch.tensor([[1.0],[2.0],[3.0]])
# yhat = lr(x)
# print("The prediction: ", yhat)
# -->
# 

# <!--Empty Space for separating topics-->
# 

# <h2 id="Cust">Build Custom Modules</h2>
# 

# Now, let's build a custom module. We can make more complex models by using this method later on. 
# 

# First, import the following library.
# 

# In[16]:


# Library for this section

from torch import nn


# Now, let us define the class: 
# 

# In[17]:


# Customize Linear Regression Class

class LR(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out


# Create an object by using the constructor. Print out the parameters we get and the model.
# 

# In[18]:


# Create the linear regression model. Print out the parameters.

lr = LR(1, 1)
print("The parameters: ", list(lr.parameters()))
print("Linear model: ", lr.linear)


# <!--Empty Space for separating topics-->
# 

# Let us try to make a prediction of a single input sample.
# 

# In[19]:


# Try our customize linear regression model with single input

x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)


# <!--Empty Space for separating topics-->
# 

# Now, let us try another example with multiple samples.
# 

# In[20]:


# Try our customize linear regression model with multiple input

x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)


# the parameters are also stored in an ordered dictionary :
# 

# In[21]:


print("Python dictionary: ", lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())


# <!--Empty Space for separating topics-->
# 

# <h3>Practice</h3>
# 

# Create an object <code>lr1</code> from the class we created before and make a prediction by using the following tensor: 
# 

# In[23]:


# Practice: Use the LR class to create a model and make a prediction of the following tensor.

x = torch.tensor([[1.0], [2.0], [3.0]])
lr(x)


# Double-click <b>here</b> for the solution.
# 
# <!-- Your answer is below:
# x=torch.tensor([[1.0],[2.0],[3.0]])
# lr1=LR(1,1)
# yhat=lr(x)
# yhat
# -->
# 

#  <!-- Your answer is below:
# x=torch.tensor([[1.0],[2.0],[3.0]])
# lr1=LR(1,1)
# yhat=lr1(x)
# yhat
# -->
# 

# <a href="https://dataplatform.cloud.ibm.com/registration/stepone?context=cpdaas&apps=data_science_experience,watson_machine_learning"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/Template/module%201/images/Watson_Studio.png"/></a>
# 

# <h2>About the Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 

# Other contributors: <a href="https://www.linkedin.com/in/michelleccarey/">Michelle Carey</a>, <a href="www.linkedin.com/in/jiahui-mavis-zhou-a4537814a">Mavis Zhou</a> 
# 

# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By | Change Description                                          |
# | ----------------- | ------- | ---------- | ----------------------------------------------------------- |
# | 2020-09-21        | 2.0     | Shubham    | Migrated Lab to Markdown and added to course repo in GitLab |
# 

# <hr>
# 

# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
