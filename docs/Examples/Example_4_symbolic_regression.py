#!/usr/bin/env python
# coding: utf-8

# # Example 4: Symbolic Regression

# The symbolic space is very dense, which means getting the correct symbolic formula (if existing at all) is a hard task. We will show how sentitive symbolic regression is, especially in the presence of noise. This is good or bad:
# 
# **Good**: One can easily find symbolic formulas that match with data quite well (within some tolerable epsilon). When one does not care about the exact symbolic formula, they might be happy with these approximate symbolic formulas that fit their data well. These approximate symbolic formulas provide some level of insight, have predictive power and are easy to compute.
# 
# **Bad**: It's hard to find the exact formula. When one does care about the exact formula, we either care about (i) its generalizability in future cases (like Newton's gravity), or (ii) fitting the clean data or solving a PDE as precise as machine precision. For case (i), it is open-ended and requires case-by-case analysis. For case (ii), we can get a (hopefully) clear signal of the correctness of a symbolic formula by noticing the loss to go down to near machine precision. We will use an example to demonstrate this below.

# ## Part I: Automated vs manual symbolic regression (How can we know that we get the exact formula?)

# In[1]:


from kan import *
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,5,1], grid=5, k=3, seed=0)

# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)
dataset['train_input'].shape, dataset['train_label'].shape


# In[2]:


# train the model
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);


# In[3]:


model = model.prune()
model(dataset['train_input'])
model.plot()


# In[4]:


# sin appears at the top of the suggestion list, which is good!
model.suggest_symbolic(0,0,0)


# In[5]:


# x^2 appears in the suggestion list (usually not top 1), but it is fine!
model.suggest_symbolic(0,1,0)


# In[6]:


# exp not even appears in the list (but note how high correlation of all these functions), which is sad!
model.suggest_symbolic(1,0,0)


# In[7]:


# let's try suggesting more by changing topk. Exp should appear in the list
# But it's very unclear why should we prefer exp over others. All of them have quite high correlation with the learned spline.
model.suggest_symbolic(1,0,0,topk=15)


# Let's train more! The loss goes down and the splines should be more exact

# In[8]:


model.train(dataset, opt="LBFGS", steps=20);
model.plot()


# In[9]:


# sin appears at the top of the suggestion list, which is good!
model.suggest_symbolic(0,0,0)


# In[10]:


# x^2 appears at the top of the suggestion list, which is good!
# But note how competitive cosh and gaussian are. They are also locally quadratic.
model.suggest_symbolic(0,1,0)


# In[11]:


# exp appears at the top of the suggestion list, which is good!
model.suggest_symbolic(1,0,0)


# The takeaway is that symbolic regression is very sensitive to noise, so if we want to extract exact symbolic formulas from trained networks, the networks need to be trained to quite high accuracy!

# In[12]:


# now let's replace every activation function with its top 1 symbolic suggestion. This is implmented in auto_symbolic()
model.auto_symbolic()

# if the user wants to constrain the symbolic space, they can pass in their symbolic libarary
# lib = ['sin', 'x^2', 'exp']
# model.auto_symbolic(lib=lib)


# After retraining, we get (almost) machine precision! This is the winning signal that this formula is (very likely to be) exact!

# In[13]:


model.train(dataset, opt="LBFGS", steps=20);
model.plot()


# In[14]:


# obtaining symbolic formula
formula, variables = model.symbolic_formula()
formula[0]


# In[15]:


# if you want to rename your variables, you could use the "var" argument
formula, variables = model.symbolic_formula(var=['\\alpha','y'])
formula[0]


# In[16]:


# one can even postprocess the formula (e.g., taking derivatives)
from sympy import *
diff(formula[0], variables[0])


# When do we know the formula we guessed is wrong (not exact)? If the data is clean (no noise), we should see the training loss does not reach machine precision

# In[17]:


# let's replace (0,1,0) with cosh
model.fix_symbolic(0,1,0,'cosh')


# In[18]:


# this loss is stuck at around 1e-3 RMSE, which is good, but not machine precision.
model.train(dataset, opt="LBFGS", steps=20);
model.plot()


# ## Part II: How hard (ill-defined) is symbolic regression, really?
# 
# In part I, we show how people can use KANs for symbolic regression, but caveat that we need to train KANs to quite high precision. This is not a problem specific to KANs though; this issue originates from symbolic regression. The space of symbolic formulas is actually quite dense, so tiny noise can make one symbolic formula transit to another. 

# ### 1D example: Adding noise to a bounded region sine

# In[19]:


def toy(bound=1., noise=0., fun=lambda x: torch.sin(torch.pi*x)):

    num_pts = 101
    x = torch.linspace(-bound,bound,steps=num_pts)
    x = x[:,None]
    y = fun(x) + torch.normal(0,1,size=(num_pts,)) * noise
    dataset = {}
    dataset['train_input'] = dataset['test_input'] = x
    dataset['train_label'] = dataset['test_label'] = y
    model = KAN(width=[1,1], grid=5, k=3, seed=0, grid_range=(-bound,bound))
    model.train(dataset, opt="LBFGS", steps=20)
    model.suggest_symbolic(0,0,0)
    model.plot()


# In[20]:


# when the function is whole range "bound=1."" (captures a whole period of sine) and has zero noise "noise=0."
# it is quite clear the function is clear
toy()


# In[21]:


# even with large noise, sine can be revealed, yeah!
toy(noise=1.)


# In[22]:


# but when bound is small and there is noise, it starts to screw up (at least becomes less clear why we should prefer sine)
toy(bound = 0.1, noise=0.1)


# ### Phase diagram of symbolic regression (how fratcal/chaotic is my phase diagram?)

# #### mix three functions $f_1(x)={\rm sin}(x)$, $f_2(x)=x^2$, and $f_3(x)={\rm exp}(x)$ such that $f(x)=af_1(x)+bf_2(x)+(1-a-b)f_3(x)$. Symbolically regress $f(x)$.

# In[23]:


def mix(a, b, bound=1):
    num_pts = 101
    x = torch.linspace(-bound,bound,steps=num_pts)
    x = x[:,None]
    y = a * torch.sin(x) + b * x**2 + (1-a-b) * torch.exp(x)
    dataset = {}
    dataset['train_input'] = dataset['test_input'] = x
    dataset['train_label'] = dataset['test_label'] = y
    model = KAN(width=[1,1], grid=10, k=3, seed=0, grid_range=(-bound,bound))
    model.train(dataset, opt="LBFGS", steps=20)
    return model.suggest_symbolic(0,0,0)[0]
    


# In[24]:


mix(a=0.2, b=0.0)


# In[25]:


# let's do a phase diagram, which looks quite "fractal"
num = 11
a_arr = np.linspace(0,1,num=num)
b_arr = np.linspace(0,1,num=num)
sf_mat = np.empty((num,num), dtype='U8')

for i in range(num):
    for j in range(num):
        a = a_arr[i]; b = b_arr[j]
        sf_mat[i,j] = mix(a, b)


# In[26]:


classes = list(set(sf_mat.reshape(-1,)))
n_class = len(classes)

colors = np.random.rand(n_class,4)
dic = {}
for i in range(n_class):
    dic[classes[i]] = colors[i]
    

img = np.zeros((num,num,4))
for i in range(num):
    for j in range(num):
        img[i][j] = dic[sf_mat[i][j]]
plt.imshow(img)


# Does this mean symbolic regression is screwed? The hope is that by incorporating reasonable inductive biases (hence reducing the symbolic search space), SR will become more robust.

# In[27]:


# we have used the default symbolic library whch contains the following functions
SYMBOLIC_LIB.keys()


# In[28]:


# we may constrain to a smaller library (pass as parameter "lib=lib" in suggest_symbolic)
lib = ['exp', 'x^2', 'sin']
def mix(a, b, bound=1):
    num_pts = 101
    x = torch.linspace(-bound,bound,steps=num_pts)
    x = x[:,None]
    y = a * torch.sin(x) + b * x**2 + (1-a-b) * torch.exp(x)
    dataset = {}
    dataset['train_input'] = dataset['test_input'] = x
    dataset['train_label'] = dataset['test_label'] = y
    model = KAN(width=[1,1], grid=10, k=3, seed=0, grid_range=(-bound,bound))
    model.train(dataset, opt="LBFGS", steps=20)
    return model.suggest_symbolic(0,0,0,lib=lib)[0]


# In[29]:


# we can redo the analysis for a more contrained (bound) region. The phase diagram becomes even more "fractal"
num = 11
a_arr = np.linspace(0,1,num=num)
b_arr = np.linspace(0,1,num=num)
sf_mat = np.empty((num,num), dtype='U8')

for i in range(num):
    for j in range(num):
        a = a_arr[i]; b = b_arr[j]
        sf_mat[i,j] = mix(a, b, bound=0.3)


# In[30]:


classes = list(set(sf_mat.reshape(-1,)))
n_class = len(classes)

colors = np.random.rand(n_class,4)
dic = {}
for i in range(n_class):
    dic[classes[i]] = colors[i]
    

img = np.zeros((num,num,4))
for i in range(num):
    for j in range(num):
        img[i][j] = dic[sf_mat[i][j]]
plt.imshow(img)

