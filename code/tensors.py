"""
Tensors operations.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotting_utils as plu
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
matplotlib.use('macOSX')


# Check type of available devices
torch.accelerator.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device=device)


##


# Tensors

# Create tensors    
x = torch.tensor(
    [[1, 2], [3, 4]],
    dtype=torch.float16, 
    device=device,
    requires_grad=True,
)       
x = torch.tensor(np.array([[[1,2],[3,4],[5,6],[7,8]]]))
x = torch.from_numpy(np.array([[[1,2,1],[3,4,1]],[[5,6,0],[7,8,0]]]))
x = x.to(device, dtype=torch.float32)
x = x.clone()
x = torch.ones((5))
x = torch.zeros((5))
x = torch.ones_like(x)
x = torch.rand((5))
x = torch.randn((5,5))
x = torch.arange(0, 10, 2)
x = torch.linspace(0, 1, steps=5)
x = torch.eye(3, 3)
x = torch.empty((2,3))
x = torch.empty_like(x)
x = torch.full((2,3), 5.0)

# Back to list and numpy arrays
x.numpy()
x.tolist()

# Indexing, slicing, setting, reshaping
x = torch.tensor(np.array([[1,2,3,4],[5,6,7,8]]))
x.reshape((2,2,2))
x[:,1:4]
x[:,[0,2]]
x[:,np.array([0,2])]
x[:,1:4] = torch.tensor([[10,11,12],[13,14,15]])

# Boolean operations and indexing
x = torch.tensor([True,False,True,False])
y = torch.tensor([True,True,False,False])
x == y
x != y
x | y
~(x == y)
torch.logical_and(x, y)
x.bitwise_and(y)
torch.logical_or(x, y)
x.bitwise_or(y)
x = torch.tensor([[1,2,3,4],[5,6,7,8]])
x[x>2]
x[:,torch.tensor([True,False,True,False])]
torch.where(x>2)
torch.argwhere(x>2)
torch.where(x>10, x, torch.nan)
x[x.nonzero()]

# Squeezing, unsqueezing, stacking, concatenating
torch.tensor(x.numpy()[np.newaxis,:]).squeeze(0)
x.unsqueeze(0).numpy() 

# Concatenate 
torch.cat([
    torch.tensor([1,2,3,4]).unsqueeze(0),
    torch.tensor([5,6,7,8]).unsqueeze(0)], 
    dim=0
)
torch.hstack([torch.tensor([1,2,3,4]), torch.tensor([5,6,7,8])])

# Algebra and matrix operations
x = torch.tensor([[1,2,3],[4,5,6]]).reshape((3,2))
y = torch.ones_like(x) * 2
x + y
y - x  
x * y
x / y
x ** 2
x ** 3
x * 0.5
x.add(y)
y.sub(x)  
x.mul(y)
x.div(y)
x.square()
x.pow(3)
x.sqrt()
x.abs()
x.square()

# Matrix operations
x.T
x.T.matmul(y)
x.T @ y
x[0,:].unsqueeze(1).T @ x[0,:].unsqueeze(1)
x[0,:].unsqueeze(1) @ x[0,:].unsqueeze(1).T
x_ = x[:-1,:].to(dtype=torch.float32)
x_.det()
x_.inverse()
x_.pinverse()
x_.matrix_exp()
x_.svd()

# Sparse 
x = torch.zeros((2,5))
x[1,2] = 1
x[1,0] = 1
x.to_sparse()

# Reduction and statistics
x.sum(dim=0)
x.mean(dim=0)
x.median(dim=0).values
x.std(dim=0)
x.var(dim=0)
x.min(dim=0).values
x.max(dim=0).values
x.argmin(dim=1)
x.argmax(dim=1)
x.cumsum(dim=1)
x.cumprod(dim=1)

# Random sampling and shuffling
np.random.seed(42)
np.random.randn(1,3)

x = np.array([0,1,2,3])
np.random.seed(42)
np.random.shuffle(x) # in place
np.random.permutation(x) # returns

np.random.seed(42)
np.random.choice(x, size=2, replace=False)

torch.manual_seed(42)
x = torch.randn(3)

torch.manual_seed(42)
x[torch.randperm(x.shape[0])]

# Gradient tracking: 1D
x = torch.tensor([2], dtype=torch.float32, requires_grad=True)
x.requires_grad
x.is_leaf
x.grad

y = (1/x + torch.exp(x)) 
k = torch.log(y) + y
z = torch.log(k)
z.requires_grad
z.is_leaf
z.grad_fn
z.grad
z.backward()
z.grad
x.grad

# Gradient tracking: 2D
x = torch.tensor([1,1], dtype=torch.float32, requires_grad=True)
x.requires_grad
x.is_leaf
x.grad

A = torch.tensor([[1,1],[2,3]], dtype=torch.float32)
B = torch.tensor([1,1], dtype=torch.float32)
A.shape
x.shape
y = A @ x
B.shape
y.shape
z = B @ y
z.shape
z.requires_grad
z.is_leaf
z.grad_fn
z.grad
z.backward()
z.grad
x.grad


##