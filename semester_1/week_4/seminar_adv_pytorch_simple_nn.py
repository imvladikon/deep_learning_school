#!/usr/bin/env python
# coding: utf-8

# <p style="align: center;"><img align=center src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" width=500 height=450/></p>
# 
# <h3 style="text-align: center;"><b>"Глубокое обучение". Продвинутый поток</b></h3>
# 
# <h2 style="text-align: center;"><b>Семинар 6. PyTorch. Создание и обучение нейронных сетей </b></h2>
# 

# # Neural networks training using PyTorch
# 
# In this notebook we build and train simple neural network using PyTorch. Our goal is to show the basics of torch framework and achieve simple understandings of how problemsolving with neural networks looks like.

# In[ ]:


import torch


# In[ ]:


device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
device


# ## Dataset Preparation
# 
# Here we learn some basic data preparation functions and classes from PyTorch as:
# 
# - `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`
# - dataset downloading api
# 
# First we would like to define our problem. Here we load the MNIST dataset (with PyTorch API)
# 
# MNIST:
# - 28x28 images of `0`, `1`, .. `9`
# - each pixel is grayscaled (float value in [0, 1))
# - targets are int values in [0, 9] (10 classes)
# - objects are ($x_i$, $y_i$), where $x_i$ shape is (1, 28, 28), $y_i$ is int value
# 
# For our purposes we will flatten the input ($x$), so our data during train will have shapes:
# 
# - `x_batch` shape `(batch_size, 784)`
# - `y_batch` shape `(batch_size)`
# 
# 

# In[ ]:


from torchvision.datasets import MNIST
import torchvision.transforms as tfs


# In[ ]:


data_tfs = tfs.Compose([
  tfs.ToTensor(),
  tfs.Normalize((0.5), (0.5))
])


# In[ ]:


# install for train and test
root = './'
train = MNIST(root, train=True,  transform=data_tfs, download=True)
test  = MNIST(root, train=False, transform=data_tfs, download=True)


# In[ ]:


print(f'Data size:\n\t train {len(train)},\n\t test {len(test)}')
print(f'Data shape:\n\t features {train[0][0].shape},\n\t target {type(test[0][1])}')


# New thing we don't need to make batch loader by ourselves. Let us use the torch implementation of it called `DataLoader` from `torch.utils.data`

# In[ ]:


from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, drop_last=True)


# In[ ]:


x_batch, y_batch = next(iter(train_loader))
x_batch.shape, y_batch.shape


# Finally we're prepared our data, so let's build some model to classify the images!

# ## Model and Train (of abnormal people)
# 
# We now how the torch computes the gradient functions during the computation of expression. Using `.backward()` method on expression **we pass the tree of gradient computing till the leafs** which are reliable for parameters of our model.
# 
# Thus, we use this knowledge to find the optimal weights of some model, which is represented by some expression.
# 
# Assume,
# 1. We want to learn linear model
# 2. For each class we use own weights to calculate logits
# 3. We use softmax on logits for probabilities of each class
# 4. Train on batches using sgd

# In[ ]:


features = 784
classes = 10


# In[ ]:


W = torch.FloatTensor(features, classes).uniform_(-1, 1) / features**0.5
W.requires_grad_()


# SGD train loop

# In[ ]:


epochs = 3
lr=1e-2
history = []


# In[ ]:


import numpy as np
from torch.nn.functional import cross_entropy


# In[ ]:


for i in range(epochs):
  for x_batch, y_batch in train_loader:
    # load batches of data correctly
    x_batch = x_batch.reshape(x_batch.shape[0], -1)

    # compute loss (log loss a.k.a. cross entropy)
    logits = x_batch @ W
    probabilities = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdims=True)

    loss = -torch.log(probabilities[range(batch_size), y_batch]).mean()
    history.append(loss.item())

    # calc gradients
    loss.backward()

    # step of gradient descent
    grad = W.grad
    with torch.no_grad():
      W -= lr * grad
    W.grad.zero_()

  print(f'{i+1},\t loss: {history[-1]}')


# And of course we should plot the loss through our training

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(10, 7))

plt.plot(history)

plt.title('Loss by batch iterations')
plt.ylabel('Entropy Loss')
plt.xlabel('batches')

plt.show()


# Some quality metrics for our linear model

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


acc = 0
batches = 0

for x_batch, y_batch in test_loader:
  # load batches of data correctly
  batches += 1
  x_batch = x_batch.view(x_batch.shape[0], -1)
  y_batch = y_batch

  preds = torch.argmax(x_batch @ W, dim=1)
  acc += (preds==y_batch).cpu().numpy().mean()

print(f'Test accuracy {acc / batches:.3}')


# Result, we now see that every variable or tensor is provided with its own `grad` and `grad_fn`. This part showed we can directly manipulate with them. Also it is very inconvinient, but sometimes we should be able to access such flexibility.
# 
# Of course PyTorch has higher level of operations between weights and grads, than we saw above. Next part introduces high level of modeling and training neural networks. 
# 

# ## Model
# 
# PyTorch is flexible framework to build any kind of neural network
# 
# Here is a table of comparison:
# 
# ```
# | API             | Flexibility | Convenience |,
# |-----------------|-------------|-------------|,
# | Barebone        | High        | Low         |,
# | `nn.Module`     | High        | Medium      |,
# | `nn.Sequential` | Low         | High        |
# ```

# 1. barebone is the approach where we directly manipulate with ternsors. In the example above we had the objective function directly expressid by weights. If we develop this method into classes we would get this level of API. **On this level we code modules by ourselves**
# 
# 2. [`nn.Module`](https://pytorch.org/docs/stable/nn.html) is the parent class for many PyTorch introduced modules. There are plenty of them. They are pretty enough to use them out-of-the-box with required parameters. Mostly we use:
# 
# - `nn.Linear`
# - `nn.Softmax`, `nn.LogSoftmax`
# - `nn.ReLU`, `nn.ELU`, `nn.LeakyReLU`
# - `nn.Tanh`, `nn.Sigmoid`
# - `nn.LSTM`, `nn.GRU`
# - `nn.Conv1d`, `nn.Conv2d`
# - `nn.MaxPool1d`, `nn.AdaptiveMaxPool1d` and others pooling
# - `nn.BatchNorm1d`, `nn.BatchNorm2d`
# - `nn.Dropout`
# - losses: `nn.CrossEntropyLoss`, `nn.NLLLoss`, `nn.MSELoss`
# - etc
# 
# 3. `nn.Sequential` is no more than sequence of different modules based on `nn.Module`. They are initiatiated by a list of modules, where output from one module goes as input for next in sequence.
# 
# 
# Let's **develop a simple sequential** to classify MNIST using **two linear layers** model.

# In[ ]:


import torch.nn as nn
from torchsummary import summary


# In[ ]:


model = nn.Sequential(
  nn.Linear(features, 64),
  nn.ReLU(),
  nn.Linear(64, classes)
)

# It is the same as:
#
# model = nn.Sequential()
# model.add_module('0', nn.Linear(features, 64))
# model.add_module('1', nn.Linear(64, classes))

model.to(device)


# We can check if everything is fine with hidden layers by `torchsummary.summary`. It needs the shape of input data to produce visualisation of model

# In[ ]:


summary(model, (features,), batch_size=228)


# ## Train
# 
# As we know, most important thing to do in solving problem are **3 things**:
# 
# 1. Model
# 2. Objective (loss function)
# 3. Optimizing (objective w.r.t. model parameters)
# 
# Good, everything we must do when using PyTorch is **define these 3 things**:
# 
# 1. Model: from `nn.Module` API
# 2. Loss: again, `nn.Module` or [`nn.functional`](https://pytorch.org/docs/stable/nn.functional.html) API
# 3. Optimizer: based on [`torch.optim.Optimizer`](https://pytorch.org/docs/stable/optim.html)
# 
# In the previous tasks when it came to optimize objective we used **direct solution or gradient descent optimizations**
# 
# Of course, there are plenty upgrades of gradient descent. It can use **adaptive step value**, **previous step gradients** and others. PyTorch also provides some classes for gradient optimizations. They are initializing with parameters they should tune for better loss value and during descent they do the step in gradient-based descent method.
# 
# Here is the most used optimizers, based on which descent algorithm is used:
# 
# - `torch.optim.Adam` uses both second and first momentum of gradient, very popular for its speed of convergence, simplicity. [paper](https://arxiv.org/abs/1412.6980)
# 
# - `torch.optim.SGD` - good-old stochastic gradient descent. Can be used with Nesterov momentum optimization
# 
# - `torch.optim.Adagrad` - [paper](https://jmlr.org/papers/v12/duchi11a.html)
# 
# - `torch.optim.RMSprop` - introduced in [slides](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
# 
# 

# First define the last 2 things: criterion and optimizer

# In[ ]:


criterion = nn.CrossEntropyLoss()      # (logsoftmax + negative likelihood) in its core, applied to logits

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))


# Finally, we are heading to **train loop**!
# 
# Previously we iterated over epochs and batches (this is called train loop). So, let's do it here

# In[ ]:


epochs = 3
history = []


# In[ ]:


for i in range(epochs):
  for x_batch, y_batch in train_loader:
    # 1. load batches of data correctly
    x_batch = x_batch.view(x_batch.shape[0], -1).to(device)
    y_batch = y_batch.to(device)

    # 2. compute scores with .forward or .__call__
    logits = model(x_batch)

    # 3. compute loss
    loss = criterion(logits, y_batch)
    history.append(loss.item())

    # 4. calc gradients
    optimizer.zero_grad()
    loss.backward()

    # 5. step of gradient descent
    optimizer.step()

  print(f'{i+1},\t loss: {history[-1]}')


# Plot and accurracy just to check the correctness.

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(10, 7))

plt.plot(history)

plt.title('Loss by batch iterations')
plt.ylabel('Entropy Loss')
plt.xlabel('batches')

plt.show()


# Some quality metrics for our two layer model

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


acc = 0
batches = 0

for x_batch, y_batch in test_loader:
  # load batch of data correctly
  batches += 1
  x_batch = x_batch.view(x_batch.shape[0], -1).to(device)
  y_batch = y_batch.to(device)

  preds = torch.argmax(model(x_batch), dim=1)
  acc += (preds==y_batch).cpu().numpy().mean()

print(f'Test accuracy {acc / batches:.3}')


# ## Summary
# 
# We have build and trained the most simple neural network. It has been done by defining:
# 
# 1. `model`
# 2. `criterion`
# 1. `optimizer`
# 
# Remember the steps during train on batch:
# 
# 1. load batch
# 2. do forward pass of model (get `scores`)
# 3. calculate `loss` (using criterion over `scores` and true labels of batch)
# 4. perform `loss.backward()` (compute gradients of loss w.r.t. parameters)
# 5. do optimization step (`optimizer.step()`)
# * zero gradients (place it everywhere but not between 4 and 5)
# * validating (after each epoch)
# 
# 

# In[ ]:




