#!/usr/bin/env python
# coding: utf-8

# # Example 9: Singularity

# Let's construct a dataset which contains singularity $f(x,y)=sin(log(x)+log(y))
#  (x>0,y>0)$

# In[1]:


from kan import KAN, create_dataset, SYMBOLIC_LIB, add_symbolic
import torch

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,1,1], grid=20, k=3, seed=0)
f = lambda x: torch.sin(2*(torch.log(x[:,[0]])+torch.log(x[:,[1]])))
dataset = create_dataset(f, n_var=2, ranges=[0.2,5])

# train the model
model.train(dataset, opt="LBFGS", steps=20);


# In[2]:


model.plot()


# In[3]:


model.fix_symbolic(0,0,0,'log')
model.fix_symbolic(0,1,0,'log')
model.fix_symbolic(1,0,0,'sin')


# In[4]:


model.train(dataset, opt="LBFGS", steps=20);


# In[5]:


model.symbolic_formula()[0][0]


# We were lucky -- singularity does not seem to be a problem in this case. But let's instead consider $f(x,y)=\sqrt{x^2+y^2}$. $x=y=0$ is a singularity point.

# In[6]:


from kan import KAN, create_dataset, SYMBOLIC_LIB, add_symbolic
import torch

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,5,1], grid=5, k=3, seed=1)
f = lambda x: torch.sqrt(x[:,[0]]**2+x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)

# train the model
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);


# In[7]:


model.plot()


# In[8]:


model = model.prune()
model(dataset['train_input'])
model.plot()


# In[9]:


model.train(dataset, opt="LBFGS", steps=20);


# In[10]:


model.plot()


# In[11]:


model.auto_symbolic()


# In[12]:


model.symbolic_formula()[0][0]


# In[13]:


# will give nan, it's a bug that should be resolved later. 
# But happy to see the above already give a formula that is close enough to ground truth
model.train(dataset, opt="LBFGS", steps=20, lr=1e-3, update_grid=False);

