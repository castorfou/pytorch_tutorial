#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/Template/module%201/images/IDSNlogo.png" width="300" alt="cognitiveclass.ai logo"  />
# </center>
# 

# <h2>Objective</h2><ul><li> How to make a prediction using multiple samples.</li></ul> 
# 

# # Table of Contents
# 
# In this lab, we will  review how to make a prediction for Linear Regression with Multiple Output. 
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <li><a href="#ref2">Build Custom Modules </a></li>
# 
# <br>
# <p></p>
# Estimated Time Needed: <strong>15 min</strong>
# </div>
# 
# <hr>
# 

# <a id="ref1"></a>
# 
# <h2 align=center>Class Linear  </h2>
# 

# In[1]:


from torch import nn
import torch


# Set the random seed:
# 

# In[2]:


torch.manual_seed(1)


# Set the random seed:
# 

# In[3]:


class linear_regression(nn.Module):
    def __init__(self,input_size,output_size):
        super(linear_regression,self).__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x):
        yhat=self.linear(x)
        return yhat


# create a linear regression  object, as our input and output will be two we set the parameters accordingly 
# 

# In[4]:


model=linear_regression(1,10)
model(torch.tensor([1.0]))


# we can use the diagram to represent the model or object 
# 

# <img src = "https://ibm.box.com/shared/static/icmwnxru7nytlhnq5x486rffea9ncpk7.png" width = 600, align = "center">
# 

# we can see the parameters 
# 

# In[5]:


list(model.parameters())


# we can create a tensor with two rows representing one sample of data
# 

# In[6]:


x=torch.tensor([[1.0]])


# we can make a prediction 
# 

# In[7]:


yhat=model(x)
yhat


# each row in the following tensor represents a different sample 
# 

# In[8]:


X=torch.tensor([[1.0],[1.0],[3.0]])


# we can make a prediction using multiple samples 
# 

# In[9]:


Yhat=model(X)
Yhat


# the following figure represents the operation, where the red and blue  represents the different parameters, and the different shades of green represent  different samples.
# 

#  <img src = "https://ibm.box.com/shared/static/768cul6pj8hc93uh9ujpajihnp8xdukx.png" width = 600, align = "center">
# 

# <a href="https://dataplatform.cloud.ibm.com/registration/stepone?context=cpdaas&apps=data_science_experience,watson_machine_learning"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/Template/module%201/images/Watson_Studio.png"/></a>
# 

# # About the Authors:
# 
#  [Joseph Santarcangelo](https://www.linkedin.com/in/joseph-s-50398b136/?utm_email=Email&utm_source=Nurture&utm_content=000026UJ&utm_term=10006555&utm_campaign=PLACEHOLDER&utm_id=SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork-20647811) has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 
# Other contributors: [Michelle Carey](https://www.linkedin.com/in/michelleccarey/?utm_email=Email&utm_source=Nurture&utm_content=000026UJ&utm_term=10006555&utm_campaign=PLACEHOLDER&utm_id=SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork-20647811) 
# 

# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By | Change Description                                          |
# | ----------------- | ------- | ---------- | ----------------------------------------------------------- |
# | 2020-09-23        | 2.0     | Shubham    | Migrated Lab to Markdown and added to course repo in GitLab |
# 

# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
