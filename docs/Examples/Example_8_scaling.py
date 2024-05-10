#!/usr/bin/env python
# coding: utf-8

# # Example 8: KANs' Scaling Laws

# In this example, we show KAN's scaling laws (wrt model params and data size)

# In[1]:


from kan import *

# initialize KAN with G=3
model = KAN(width=[2,1,1], grid=3, k=3)

data_sizes = np.array([100,300,1000,3000])
grids = np.array([5,10,20,50,100])

train_losses = np.zeros((data_sizes.shape[0], grids.shape[0]))
test_losses = np.zeros((data_sizes.shape[0], grids.shape[0]))
steps = 50
k = 3

for j in range(data_sizes.shape[0]):
    data_size = data_sizes[j]
    print(f'data_size={data_size}')
    # create dataset
    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(f, n_var=2, train_num=data_size)
    
    for i in range(grids.shape[0]):
        print(f'grid_size={grids[i]}')
        if i == 0:
            model = KAN(width=[2,1,1], grid=grids[i], k=k)
        if i != 0:
            model = KAN(width=[2,1,1], grid=grids[i], k=k).initialize_from_another_model(model, dataset['train_input'])
        results = model.train(dataset, opt="LBFGS", steps=steps, stop_grid_update_step = 30)
        train_losses[j][i] = results['train_loss'][-1]
        test_losses[j][i] = results['test_loss'][-1]


# Fix data size, study model (grid) size scaling. Roughly display $N^{-4}$ scaling.

# In[2]:


for i in range(data_sizes.shape[0]):
    plt.plot(grids, train_losses[i,:], marker="o")
plt.xscale('log')
plt.yscale('log')
plt.plot(np.array([5,100]), 0.1*np.array([3,100])**(-4.), ls="--", color="black")
plt.legend([f'data={data_sizes[i]}' for i in range(data_sizes.shape[0])]+[r'$N^{-4}$'])
plt.ylabel('train RMSE')
plt.xlabel('grid size')


# In[3]:


for i in range(data_sizes.shape[0]):
    plt.plot(grids, test_losses[i,:], marker="o")
plt.xscale('log')
plt.yscale('log')
plt.plot(np.array([5,100]), 0.1*np.array([3,100])**(-4.), ls="--", color="black")
plt.legend([f'data={data_sizes[i]}' for i in range(data_sizes.shape[0])]+[r'$N^{-4}$'])
plt.ylabel('test RMSE')
plt.xlabel('grid size')


# Fix model (grid) size, study data size scaling. No clear power law scaling. But we observe that: (1) increasing data size has no harm to performance. (2) powerful model (larger grid size) can benefit more from data size increase. Ideally one would want to increase data size and model size together so that their complexity always match.

# In[4]:


for i in range(grids.shape[0]):
    plt.plot(data_sizes, train_losses[:,i], marker="o")
plt.xscale('log')
plt.yscale('log')
plt.plot(np.array([100,3000]), 1e8*np.array([100,3000])**(-4.), ls="--", color="black")
plt.legend([f'grid={grids[i]}' for i in range(grids.shape[0])]+[r'$N^{-4}$'])
plt.ylabel('train RMSE')
plt.xlabel('data size')


# In[5]:


for i in range(grids.shape[0]):
    plt.plot(data_sizes, test_losses[:,i], marker="o")
plt.xscale('log')
plt.yscale('log')
plt.plot(np.array([100,3000]), 1e5*np.array([100,3000])**(-4.), ls="--", color="black")
plt.legend([f'grid={grids[i]}' for i in range(grids.shape[0])]+[r'$N^{-4}$'])
plt.ylabel('test RMSE')
plt.xlabel('data size')

