import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import sympy

# 用于生成符号回归数据集的工具函数
# sigmoid = sympy.Function('sigmoid')
# name: (torch implementation, sympy implementation)
# 代码中的SYMBOLIC_LIB是一个字典，它包含了一些常见的符号函数及其对应的PyTorch和Sympy实现。用户可以根据需要使用这些现成的符号函数，也可以通过add_symbolic函数自定义添加新的符号函数。
SYMBOLIC_LIB = {'x': (lambda x: x, lambda x: x),
                 'x^2': (lambda x: x**2, lambda x: x**2),
                 'x^3': (lambda x: x**3, lambda x: x**3),
                 'x^4': (lambda x: x**4, lambda x: x**4),
                 '1/x': (lambda x: 1/x, lambda x: 1/x),
                 '1/x^2': (lambda x: 1/x**2, lambda x: 1/x**2),
                 '1/x^3': (lambda x: 1/x**3, lambda x: 1/x**3),
                 '1/x^4': (lambda x: 1/x**4, lambda x: 1/x**4),
                 'sqrt': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x)),
                 '1/sqrt(x)': (lambda x: 1/torch.sqrt(x), lambda x: 1/sympy.sqrt(x)),
                 'exp': (lambda x: torch.exp(x), lambda x: sympy.exp(x)),
                 'log': (lambda x: torch.log(x), lambda x: sympy.log(x)),
                 'abs': (lambda x: torch.abs(x), lambda x: sympy.Abs(x)),
                 'sin': (lambda x: torch.sin(x), lambda x: sympy.sin(x)),
                 'tan': (lambda x: torch.tan(x), lambda x: sympy.tan(x)),
                 'tanh': (lambda x: torch.tanh(x), lambda x: sympy.tanh(x)),
                 'sigmoid': (lambda x: torch.sigmoid(x), sympy.Function('sigmoid')),
                 #'relu': (lambda x: torch.relu(x), relu),
                 'sgn': (lambda x: torch.sign(x), lambda x: sympy.sign(x)),
                 'arcsin': (lambda x: torch.arcsin(x), lambda x: sympy.arcsin(x)),
                 'arctan': (lambda x: torch.arctan(x), lambda x: sympy.atan(x)),
                 'arctanh': (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x)),
                 '0': (lambda x: x*0, lambda x: x*0),
                 'gaussian': (lambda x: torch.exp(-x**2), lambda x: sympy.exp(-x**2)),
                 'cosh': (lambda x: torch.cosh(x), lambda x: sympy.cosh(x)),
                 #'logcosh': (lambda x: torch.log(torch.cosh(x)), lambda x: sympy.log(sympy.cosh(x))),
                 #'cosh^2': (lambda x: torch.cosh(x)**2, lambda x: sympy.cosh(x)**2),
}

def create_dataset(f, 
                   n_var=2, 
                   ranges = [-1,1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset
    创建符号回归数据集的函数。可以根据给定的符号函数生成训练集和测试集。
    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
         
    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    '''

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:,i] = torch.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        test_input[:,i] = torch.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        
        
    train_label = f(train_input)
    test_label = f(test_input)
        
        
    def normalize(data, mean, std):
            return (data-mean)/std
            
    if normalize_input == True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset



def fit_params(x, y, fun, a_range=(-10,10), b_range=(-10,10), grid_number=101, iteration=3, verbose=True, device='cpu'):
    '''
    fit a, b, c, d such that
    用于拟合符号函数的参数。通过给定的输入数据和目标数据，尝试找到最佳的函数参数，使得符号函数能够最好地拟合输入数据。
    .. math::
        |y-(cf(ax+b)+d)|^2
        
    is minimized. Both x and y are 1D array. Sweep a and b, find the best fitted model.
    x：输入数据的一维数组，表示自变量。
    y：目标数据的一维数组，表示因变量。
    fun：符号函数，是一个函数对象。
    a_range 和 b_range：参数 a 和 b 的搜索范围，默认为 (-10, 10)。
    grid_number：搜索范围内的步数，默认为 101。
    iteration：迭代次数，默认为 3。
    verbose：是否打印额外信息，默认为 True。
    device：指定设备，默认为 'cpu'。
    Args:
    -----
        x : 1D array
            x values
        y : 1D array
            y values
        fun : function
            symbolic function
        a_range : tuple
            sweeping range of a
        b_range : tuple
            sweeping range of b
        grid_num : int
            number of steps along a and b
        iteration : int
            number of zooming in
        verbose : bool
            print extra information if True
        device : str
            device
        
    Returns:
    --------
        a_best : float
            best fitted a
        b_best : float
            best fitted b
        c_best : float
            best fitted c
        d_best : float
            best fitted d
        r2_best : float
            best r2 (coefficient of determination)
    函数首先通过迭代搜索 a 和 b 的最佳值，以最小化损失函数 |y-(cf(ax+b)+d)|^2。然后，根据最佳的 a 和 b 值，计算符号函数 fun 的输出，拟合一个线性模型来获取参数 c 和 d。最后返回最佳参数和对应的拟合效果（用 R^2 值衡量）
    Example
    -------
    首先生成了一组输入数据 x 和相应的目标数据 y。输入数据 x 是在 -1 到 1 之间均匀分布的 100 个数据点，而目标数据 y 是根据 x 通过一个带有噪声的正弦函数生成的。
    接着，我们调用了 fit_params 函数，传入了输入数据 x、目标数据 y 和符号函数 torch.sin。这个函数的目标是通过拟合参数 a、b、c 和 d，找到一个能够最好地拟合输入数据 x 和目标数据 y 的符号函数模型。
    最后，函数返回了拟合出来的最佳参数和对应的拟合效果，其中 r2 是拟合效果的评价指标，越接近 1 表示拟合效果越好。在这个示例中，r2 的值非常接近 1，表明拟合效果非常好，而返回的参数值是 (2.9982, 1.9996, 5.0053, 0.7011)，分别对应着拟合出的 a、b、c 和 d。
    >>> num = 100
    >>> x = torch.linspace(-1,1,steps=num)
    >>> noises = torch.normal(0,1,(num,)) * 0.02
    >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
    >>> fit_params(x, y, torch.sin)
    r2 is 0.9999727010726929
    (tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))
    '''
    # fit a, b, c, d such that y=c*fun(a*x+b)+d; both x and y are 1D array.
    # sweep a and b, choose the best fitted model   
    for _ in range(iteration):
        a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number, device=device)
        b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number, device=device)
        a_grid, b_grid = torch.meshgrid(a_, b_, indexing='ij')
        post_fun = fun(a_grid[None,:,:] * x[:,None,None] + b_grid[None,:,:])
        x_mean = torch.mean(post_fun, dim=[0], keepdim=True)
        y_mean = torch.mean(y, dim=[0], keepdim=True)
        numerator = torch.sum((post_fun - x_mean)*(y-y_mean)[:,None,None], dim=0)**2
        denominator = torch.sum((post_fun - x_mean)**2, dim=0)*torch.sum((y - y_mean)[:,None,None]**2, dim=0)
        r2 = numerator/(denominator+1e-4)
        r2 = torch.nan_to_num(r2)
        
        
        best_id = torch.argmax(r2)
        a_id, b_id = torch.div(best_id, grid_number, rounding_mode='floor'), best_id % grid_number
        
        
        if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
            if _ == 0 and verbose==True:
                print('Best value at boundary.')
            if a_id == 0:
                a_arange = [a_[0], a_[1]]
            if a_id == grid_number - 1:
                a_arange = [a_[-2], a_[-1]]
            if b_id == 0:
                b_arange = [b_[0], b_[1]]
            if b_id == grid_number - 1:
                b_arange = [b_[-2], b_[-1]]
            
        else:
            a_range = [a_[a_id-1], a_[a_id+1]]
            b_range = [b_[b_id-1], b_[b_id+1]]
            
    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]
    
    if verbose == True:
        print(f"r2 is {r2_best}")
        if r2_best < 0.9:
            print(f'r2 is not very high, please double check if you are choosing the correct symbolic function.')

    post_fun = torch.nan_to_num(post_fun)
    reg = LinearRegression().fit(post_fun[:,None].detach().cpu().numpy(), y.detach().cpu().numpy())
    c_best = torch.from_numpy(reg.coef_)[0].to(device)
    d_best = torch.from_numpy(np.array(reg.intercept_)).to(device)
    return torch.stack([a_best, b_best, c_best, d_best]), r2_best



def add_symbolic(name, fun):
    '''
    add a symbolic function to library
    add_symbolic: 用于向符号库中添加新的符号函数。
    Args:
    -----
        name : str
            name of the function
        fun : fun
            torch function or lambda function
    
    Returns:
    --------
        None
    
    Example
    -------
    >>> print(SYMBOLIC_LIB['Bessel'])
    KeyError: 'Bessel'
    >>> add_symbolic('Bessel', torch.special.bessel_j0)
    >>> print(SYMBOLIC_LIB['Bessel'])
    (<built-in function special_bessel_j0>, Bessel)
    '''
    exec(f"globals()['{name}'] = sympy.Function('{name}')")
    SYMBOLIC_LIB[name] = (fun, globals()[name])
    