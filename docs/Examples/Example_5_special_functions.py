#!/usr/bin/env python
# coding: utf-8

# # Example 5: Special functions

# Let's construct a dataset which contains special functions $f(x,y)={\rm exp}(J_0(20x)+y^2)$, where $J_0(x)$ is the Bessel function.

# In[1]:


from kan import KAN, create_dataset, SYMBOLIC_LIB, add_symbolic
import torch

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,5,1], grid=20, k=3, seed=0)
f = lambda x: torch.exp(torch.special.bessel_j0(20*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)

# train the model
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);


# Plot trained KAN, the bessel function shows up in the bettom left

# In[2]:


model = model.prune()
model(dataset['train_input'])
model.plot()


# suggest_symbolic does not return anything that matches with it, since Bessel function isn't included in the default SYMBOLIC_LIB. We want to add Bessel to it.

# In[3]:


model.suggest_symbolic(0,0,0)


# In[4]:


SYMBOLIC_LIB.keys()


# In[5]:


# add bessel function J0 to the symbolic library
# we should include a name and a pytorch implementation
add_symbolic('J0', torch.special.bessel_j0)


# After adding Bessel, we check suggest_symbolic again

# In[6]:


# J0 shows up but not top 1, why?

model.suggest_symbolic(0,0,0)


# In[7]:


# This is because the ground truth is J0(20x) which involves 20 which is too large.
# our default search is in (-10,10)
# so we need to set the search range bigger in order to include 20
# now J0 appears at the top of the list

model.suggest_symbolic(0,0,0,a_range=(-40,40))


# In[8]:


model.train(dataset, opt="LBFGS", steps=20);


# In[9]:


model.plot()


# In[10]:


model.suggest_symbolic(0,0,0,a_range=(-40,40))


# Finish the rest of symbolic regression

# In[11]:


model.fix_symbolic(0,0,0,'J0',a_range=(-40,40))


# In[12]:


model.auto_symbolic()


# In[13]:


model.plot()


# In[14]:


model.train(dataset, opt="LBFGS", steps=20);


# In[15]:


model.plot()


# In[16]:


model.suggest_symbolic(1,0,0)


# In[17]:


model.fix_symbolic(1,0,0,'exp')


# In[18]:


# why can't we reach machine precision (because LBFGS early stops?)? The symbolic formula is correct though.
model.train(dataset, opt="LBFGS", steps=20);
model.symbolic_formula()[0][0]

