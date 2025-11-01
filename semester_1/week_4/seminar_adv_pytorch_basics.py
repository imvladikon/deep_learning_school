#!/usr/bin/env python
# coding: utf-8

# <p style="align: center;"><img align=center src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" width=500 height=450/></p>
# 
# <h3 style="text-align: center;"><b>"Глубокое обучение". Продвинутый поток</b></h3>
# 
# <h2 style="text-align: center;"><b>Семинар 6. Основы библиотеки PyTorch </b></h2>
# 

# # PyTorch basics: syntax, torch.cuda and torch.autograd</b></h2>

# <p style="align: center;"><img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" width=400 height=100></p>

# Hi! In this notebook we will cover the basics of the **PyTorch deep learning framework**. 

# <h3 style="text-align: center;"><b>Intro</b></h3>

# **Frameworks** are the specific code libraries with their own internal structure and pipelines.

# There are many deep learning frameworks nowadays (02/2019). The difference between them is in the internal computation principles. For example, in **[Caffe](http://caffe.berkeleyvision.org/)** and **[Caffe2](https://caffe2.ai/)** you write the code using some "ready blocks" (just like the $LEGO^{TM}$ :). In **[TensorFlow](https://www.tensorflow.org/)** and **[Theano](http://deeplearning.net/software/theano/)** you declare the computation graph at first, then compile it and use it for inference/training (`tf.session()`). By the way, now TensorFlow (since v1.10) has the [Eager Execution](https://www.tensorflow.org/guide/eager), which can be handy for fast prototyping and debugging. **[Keras](https://keras.io/)** is a very popular and useful DL framework that allows to create networks fast and has many demanding features. 

# <p style="align: center;"><img src="https://habrastorage.org/web/e3e/c3e/b78/e3ec3eb78d714a7993a6b922911c0866.png" width=500 height=500></p>  
# <p style="text-align: center;"><i>Image credit: https://habr.com/post/334380/</i><p>

# We will use PyTorch bacause it's been actively developed and supported by the community and [Facebook AI Research](https://research.fb.com/category/facebook-ai-research/).

# <h3 style="text-align: center;"><b>Installation</b></h3>

# The detailed instruction on how to install PyTorch you can find on the [official PyTorch website](https://pytorch.org/).

# ## Syntax

# In[ ]:


import torch


# Some facts about PyTorch:  
# - dynamic computation graph
# - handy `torch.nn` and `torchvision` modules for fast neural network prototyping
# - even faster than TensorFlow on some tasks
# - allows to use GPU easily

# At its core, PyTorch provides two main features:
# 
# - An n-dimensional Tensor, similar to numpy but can run on GPUs
# - Automatic differentiation for building and training neural networks

# If PyTorch was a formula, it would be:  
# 
# $$PyTorch = NumPy + CUDA + Autograd$$

# (CUDA - [wiki](https://en.wikipedia.org/wiki/CUDA))

# Let's see how we can use PyTorch to operate with vectors and tensors.  
# 
# Recall that **a tensor** is a multidimensional vector, e.g. :  
# 
# `x = np.array([1,2,3])` -- a vector = a tensor with 1 dimension (to be more precise: `(3,)`)  
# `y = np.array([[1, 2, 3], [4, 5, 6]])` -- a matrix = a tensor with 2 dimensions (`(2, 3)` in this case)  
# `z = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],  
#                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  
#                [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])` -- "a cube" (3, 3, 3) = a tensor with 3 dimensions (`(3, 3, 3)` in this case)

# One real example of 3-dimensional tensor is **an image**, it has 3 dimensions: `height`, `width` and the `channel depth` (= 3 for color images, 1 for a greyscale). You can think of it as of parallelepiped consisting of the real numbers.

# In PyTorch we will use `torch.Tensor` (`FloatTensor`, `IntTensor`, `ByteTensor`) for all the computations.

# All tensor types:

# In[ ]:


torch.HalfTensor      # 16 бит, floating point
torch.FloatTensor     # 32 бита, floating point
torch.DoubleTensor    # 64 бита, floating point

torch.ShortTensor     # 16 бит, integer, signed
torch.IntTensor       # 32 бита, integer, signed
torch.LongTensor      # 64 бита, integer, signed

torch.CharTensor      # 8 бит, integer, signed
torch.ByteTensor      # 8 бит, integer, unsigned


# We will use only `torch.FloatTensor()` and `torch.IntTensor()`. 

# Let's begin to do something!

# * Creating the tensor:

# In[ ]:


a = torch.FloatTensor([1, 2])
a


# In[ ]:


a.shape


# In[ ]:


b = torch.FloatTensor([[1,2,3], [4,5,6]])
b


# In[ ]:


b.shape


# In[ ]:


x = torch.FloatTensor(2,3,4)


# In[ ]:


x


# In[ ]:


x = torch.FloatTensor(100)
x


# In[ ]:


x = torch.IntTensor(45, 57, 14, 2)
x.shape


# **Note:** if you create `torch.Tensor` with the following constructor it will be filled with the "random trash numbers":

# In[ ]:


x = torch.IntTensor(3, 2, 4)
x


# Here is a way to fill a new tensor with zeroes:

# In[ ]:


x1 = torch.FloatTensor(3, 2, 4)
x1.zero_()
x2 = torch.zeros(3, 2, 4)
x3 = torch.zeros_like(x1)

assert torch.allclose(x1, x2) and torch.allclose(x1, x3)
x1


# Random distribution initialization

# In[ ]:


x = torch.randn((2,3))                # Normal(0, 1) with shape
x


# In[ ]:


x.random_(0, 10)                      # discrete U[0, 10]
x.uniform_(0, 1)                      # U[0, 1]
x.normal_(mean=0, std=1)              # Normal with mean and std
x.bernoulli_(p=0.5)                   # bernoulli with parameter p


# ## Numpy -> Torch
# 
# All numpy function have its pair in torch.
# 
# https://github.com/torch/torch7/wiki/Torch-for-Numpy-users

# `np.reshape()` == `torch.view()`:

# In[ ]:


b, b.stride()


# In[ ]:


b.view(3, 2), b.view(3, 2).stride()  


# **Note:** `torch.view()` creates a new tensor, one the old one remains unchanged

# In[ ]:


b.view(-1)


# In[ ]:


b


# In[ ]:


b.T.stride(), b.is_contiguous(), b.T.is_contiguous()


# In[ ]:


b.reshape(-1) # returns view or contigues tensor


# In[ ]:


b


# * Change a tensor type:

# In[ ]:


a = torch.FloatTensor([1.5, 3.2, -7])


# In[ ]:


a.type_as(torch.IntTensor())


# In[ ]:


a.to(torch.int32)


# In[ ]:


a.type_as(torch.ByteTensor())


# In[ ]:


a.to(torch.uint8)


# **Note:** `.type_as()` creates a new tensor, the old one remains unchanged

# In[ ]:


a


# * Indexing is just like in `NumPy`:

# In[ ]:


a = torch.FloatTensor([[100, 20, 35], [15, 163, 534], [52, 90, 66]])
a


# In[ ]:


a[0, 0]


# In[ ]:


a[0:2, 1]


# **Ariphmetics and boolean operations** and their analogues:  
# 
# | Operator | Analogue |
# |:-:|:-:|
# |`+`| `torch.add()` |
# |`-`| `torch.sub()` |
# |`*`| `torch.mul()` |
# |`/`| `torch.div()` |

# * Addition:

# In[ ]:


a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
b = torch.FloatTensor([[-1, -2, -3], [-10, -20, -30], [100, 200, 300]])


# In[ ]:


a + b


# In[ ]:


a.add(b)


# In[ ]:


b = -a
b


# In[ ]:


a + b


# * Subtraction:

# In[ ]:


a - b


# In[ ]:


a.sub(b) # copy


# In[ ]:


a.sub_(b) # inplace


# * Multiplication (elementwise):

# In[ ]:


a * b


# In[ ]:


a.mul(b)


# * Division (elementwise):

# In[ ]:


a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
b = torch.FloatTensor([[-1, -2, -3], [-10, -20, -30], [100, 200, 300]])


# In[ ]:


a / b


# In[ ]:


a.div(b)


# **Note:** all this operations create new tensors, the old tensors remain unchanged

# In[ ]:


a


# In[ ]:


b


# * Comparison operators:

# In[ ]:


a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
b = torch.FloatTensor([[-1, -2, -3], [-10, -20, -30], [100, 200, 300]])


# In[ ]:


a == b


# In[ ]:


a != b


# In[ ]:


a < b


# In[ ]:


a > b


# * Using boolean mask indexing:

# In[ ]:


a[a > b]


# In[ ]:


b[a == b]


# Elementwise application of the **universal functions**:

# In[ ]:


a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])


# In[ ]:


a.sin()


# In[ ]:


torch.sin(a)


# In[ ]:


a.tan()


# In[ ]:


a.exp()


# In[ ]:


a.log()


# In[ ]:


b = -a
b


# In[ ]:


b.abs()


# * The sum, mean, max, min:

# In[ ]:


a.sum(dim=1)


# In[ ]:


a.mean()


# Along axis:

# In[ ]:


a


# In[ ]:


a.sum(dim=0)


# In[ ]:


a.sum(1)


# In[ ]:


a.max()


# In[ ]:


a.max(0)


# In[ ]:


a.min()


# In[ ]:


a.min(0)


# **Note:** the second tensor returned by `.max()` and `.min()` contains the indices of max/min elements along this axis. E.g. in that case `a.min()` returned `(1, 2, 3)` which are the minimum elements along 0 axis (along columns) and their indices along 0 axis are `(0, 0, 0)`.

# **Matrix operations**:

# * Transpose a tensor:

# In[ ]:


a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
a


# In[ ]:


a.t()


# It is not not the inplace operation too:

# In[ ]:


a


# * Dot product of vectors:

# In[ ]:


a = torch.FloatTensor([1, 2, 3, 4, 5, 6])
b = torch.FloatTensor([-1, -2, -4, -6, -8, -10])


# In[ ]:


a.dot(b)


# In[ ]:


a.shape, b.shape


# In[ ]:


a @ b


# In[ ]:


type(a)


# In[ ]:


type(b)


# In[ ]:


type(a @ b)


# * Matrix product:

# In[ ]:


a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
b = torch.FloatTensor([[-1, -2, -3], [-10, -20, -30], [100, 200, 300]])


# In[ ]:


a.mm(b)


# In[ ]:


a @ b


# Remain unchanged:

# In[ ]:


a


# In[ ]:


b


# In[ ]:


a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
b = torch.FloatTensor([[-1], [-10], [100]])


# In[ ]:


print(a.shape, b.shape)


# In[ ]:


a @ b


# If we unroll the tensor `b` in an array (`torch.view(-1)`) the multiplication would be like with the column:

# In[ ]:


b


# In[ ]:


b.view(-1)


# In[ ]:


a @ b.view(-1)


# In[ ]:


a.mv(b.view(-1))


# In[ ]:


y = torch.Tensor(2, 3, 4, 5)
z = torch.Tensor(2, 3, 5, 6)
(y @ z).shape


# **From NumPu to PyTorch conversion**:

# In[ ]:


import numpy as np

a = np.random.rand(3, 3)
a


# In[ ]:


b = torch.from_numpy(a)
b


# **NOTE!** `a` and `b` have the same data storage, so the changes in one tensor will lead to the changes in another:

# In[ ]:


b -= b
b


# In[ ]:


a


# **From PyTorch to NumPy conversion:**

# In[ ]:


a = torch.FloatTensor(2, 3, 4)
a


# In[ ]:


type(a)


# In[ ]:


x = a.numpy()
x


# In[ ]:


x.shape


# In[ ]:


type(x)


# In[ ]:


x -= x


# In[ ]:


a


# Let's write the `forward_pass(X, w)` ($w_0$ is a part of the $w$) for a single neuron (activation = sigmoid) using PyTorch:

# In[ ]:


def forward_pass(X, w):
    return torch.sigmoid(X @ w)


# In[ ]:


X = torch.FloatTensor([[-5, 5], [2, 3], [1, -1]])
w = torch.FloatTensor([[-0.5], [2.5]])
result = forward_pass(X, w)
print('result: {}'.format(result))


# ## <h1 style="text-align: center;"><a href="https://ru.wikipedia.org/wiki/CUDA">CUDA</a></h3>

# [CUDA documentation](https://docs.nvidia.com/cuda/)

# We can use both CPU (Central Processing Unit) and GPU (Graphical Processing Unit) to make the computations with PyTorch. We can switch between them easily, this is one of the most important things in PyTorch framework.

# In[ ]:


x = torch.FloatTensor(1024, 10024).uniform_()
x


# In[ ]:


x.is_cuda


# Place a tensor on GPU (GPU memory is used):

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


x = x.cuda()


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


x


# In[ ]:


x = x.cpu()
get_ipython().system('nvidia-smi')

torch.cuda.empty_cache()
get_ipython().system('nvidia-smi')


# In[ ]:


device = torch.device("cuda:0")
x = x.to(device)
x


# Let's multiply two tensors on GPU and then move the result on the CPU:

# In[ ]:


a = torch.FloatTensor(10000, 10000).uniform_()
b = torch.FloatTensor(10000, 10000).uniform_()
c = a.cuda().mul(b.cuda()).cpu()


# In[ ]:


c


# In[ ]:


a


# Tensors placed on CPU and tensors placed on GPU are unavailable for each other:

# In[ ]:


a = torch.FloatTensor(10000, 10000).uniform_().cpu()
b = torch.FloatTensor(10000, 10000).uniform_().cuda()


# In[ ]:


a + b


# Example of working with GPU:

# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


x = torch.FloatTensor(5, 5, 5).uniform_()

# check for CUDA availability (NVIDIA GPU)
if torch.cuda.is_available():
    # get the CUDA device name
    device = torch.device('cuda')          # CUDA-device object
    y = torch.ones_like(x, device=device)  # create a tensor on GPU
    x = x.to(device)                       # or just `.to("cuda")`
    z = x + y
    print(z)
    # you can set the type while `.to` operation
    print(z.to("cpu", torch.double))


# ## AutoGrad

# **Chain rule (a.k.a. backpropagation in NN)** used here
# 
# Assume we have $f(w(\theta))$
# $${\frac  {\partial{f}}{\partial{\theta}}}
# ={\frac  {\partial{f}}{\partial{w}}}\cdot {\frac  {\partial{w}}{\partial{\theta}}}$$
# 
# 
# *Additional reading: In multidimentional case it is described by composition of partial derivatives:*
# $$
# D_\theta(f\circ w) = D_{w(\theta)}(f)\circ D_\theta(w)
# $$
# 
# Simple example of gradient propagation:
# 
# $$y = \sin \left(x_2^2(x_1 + x_2)\right)$$
# 
# <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0010465515004099-gr1.jpg" width=700></img>
# 

# The autograd package provides automatic differentiation for all operations on Tensors. It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different.

# The examples:

# In[ ]:


dtype = torch.float
device = torch.device("cuda:0")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 3, 3, 10

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)


# In[ ]:


y_pred = (x @ w1).clamp(min=0).matmul(w2)
loss = (y_pred - y).pow(2).sum()
# calculate the gradients
loss.backward()


# In[ ]:


print((y_pred - y).pow(2).sum())


# In[ ]:


w1.grad, w2.grad


# In[ ]:


loss.grad # can't access to non-leaf grad in AD tree


# In[ ]:


# make the variable remember grad of loss
y_pred = (x @ w1).clamp(min=0).matmul(w2)
y_pred.retain_grad()

loss = (y_pred - y).pow(2).sum()
loss.retain_grad()

loss.backward()


# In[ ]:


loss.grad


# In[ ]:


x.grad # doesn't require grad


# In[ ]:


y.grad # doesn't require grad


# **NOTE:** the gradients are placed into the `.grad` field of tensors (variables) on which gradients were calculated. Gradients *are not placed* in the variable `loss` here!

# In[ ]:


w1


# In[ ]:


with torch.no_grad():
    pass


# <h3 style="text-align: center;">Further reading:<b></b></h3>

# *1). Official PyTorch tutorials: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py*

# *2). arXiv article about the deep learning frameworks comparison: https://arxiv.org/pdf/1511.06435.pdf*

# *3). Useful repo with different tutorials: https://github.com/yunjey/pytorch-tutorial*

# *4). Facebook AI Research (main contributor of PyTorch) website: https://facebook.ai/developers/tools*
