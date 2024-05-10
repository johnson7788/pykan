import torch


def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    x: 一个 2D 的 Torch 张量，形状为 (样条数量, 样本数量)，包含了输入值。
grid: 一个 2D 的 Torch 张量，形状为 (样条数量, 网格点数量)，包含了用于评估 B-样条基函数的网格点。
k: 一个整数，表示样条的分段多项式阶数，默认值为 0。
extend: 一个布尔值，表示是否在两端扩展网格点。如果为 True，则在两端扩展 k 个点；如果为 False，则不进行扩展（零边界条件）。默认为 True。
device: 一个字符串，表示张量计算所在的设备，可以是 "cpu" 或 "cuda"。
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    如果 extend 参数为 True，则调用 extend_grid 函数在网格的两端扩展 k 个点。
将输入的 x 和网格点 grid 分别扩展为 3D 张量，以便后续计算。
根据是否存在阶数 k，计算 B-样条基函数的值。如果 k 为 0，则直接根据输入 x 与网格点的关系判断基函数的值；如果 k 大于 0，则递归调用 B_batch 函数计算 k-1 阶的基函数，然后根据 B-样条的递推关系计算当前阶的基函数值。
返回计算得到的基函数值
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.
      
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])

这段代码定义了一个名为 extend_grid 的函数，用于在 B-样条计算中在网格的两端扩展额外的点，以用于处理边界情况。

函数的参数如下：

grid: 一个 2D 的 Torch 张量，形状为 (批量大小, 网格点数量)，包含了用于 B-样条计算的原始网格点。
k_extend: 一个整数，表示要在每端扩展的网格点数，默认为 0。
函数的主要步骤是：

计算网格两端的间距 h，该间距为最后一个网格点和第一个网格点之间的距离除以网格点数量减一，以保持等距离。
在每一端依次添加 k_extend 个网格点，保持等距离。
将扩展后的网格点转换为指定的设备上，如 "cpu" 或 "cuda"。
返回扩展后的网格点。
这个函数主要用于在进行 B-样条计算时，保证边界处的插值或拟合能够得到正确的结果。
    '''

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value


def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
    >>> coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    '''
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    这段代码是用于将 B-样条曲线转换为 B-样条系数，通过最小二乘法实现。这个函数的目的是将给定的一组样本点 (x_eval, y_eval) 转换为 B-样条的系数。
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        grid : 2D torch.tensor
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        x_eval: 一个 2D 的 Torch 张量，形状为 (样条数量, 样本数量)，包含了样条的 x 坐标值。
        y_eval: 一个 2D 的 Torch 张量，形状为 (样条数量, 样本数量)，包含了样条的 y 坐标值。
        grid: 一个 2D 的 Torch 张量，形状为 (样条数量, 网格点数量)，包含了用于拟合 B-样条曲线的网格点。
        k: 一个整数，表示样条的分段多项式阶数。
        device: 一个字符串，表示张量计算所在的设备，可以是 "cpu" 或 "cuda"。
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> y_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> curve2coef(x_eval, y_eval, grids, k=k).shape
    torch.Size([5, 13])
    '''
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    # 使用 Torch 提供的最小二乘函数 torch.linalg.lstsq 对得到的系数矩阵进行最小二乘拟合，拟合目标是样本 y 坐标值。这一步返回的结果是最小二乘解，形状为 (样条数量, 网格点数量, 1)。
    coef = torch.linalg.lstsq(mat.to('cpu'), y_eval.unsqueeze(dim=2).to('cpu')).solution[:, :, 0]  # sometimes 'cuda' version may diverge
    return coef.to(device)
