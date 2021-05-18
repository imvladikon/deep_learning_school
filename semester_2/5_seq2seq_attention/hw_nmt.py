#!/usr/bin/env python
# coding: utf-8

# <img src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" width=500, height=450>
# <h3 style="text-align: center;"><b>Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ</b></h3>

# https://github.com/ManhPP/NMT_ATT/blob/b05d8c40789f4b10a5349492e969ceca92103b1a/model.py

# ***Some parts of the notebook are almost the exact copy of***  https://github.com/yandexdataschool/nlp_course

# ##  Attention
# 
# Attention layer can take in the previous hidden state of the decoder $s_{t-1}$, and all of the stacked forward and backward hidden states $H$ from the encoder. The layer will output an attention vector $a_t$, that is the length of the source sentence, each element is between 0 and 1 and the entire vector sums to 1.
# 
# Intuitively, this layer takes what we have decoded so far $s_{t-1}$, and all of what we have encoded $H$, to produce a vector $a_t$, that represents which words in the source sentence we should pay the most attention to in order to correctly predict the next word to decode $\hat{y}_{t+1}$. The decoder input word that has been embedded  $y_t$.
# 
# You can use any type of the attention scores between previous hidden state of the encoder $s_{t-1}$ and hidden state of the decoder $h \in H$, you prefer. We have met at least three of them:<br><br>
# 
# $$\operatorname{score}\left(\boldsymbol{h}, \boldsymbol{s}_{t-1}\right)=\left\{\begin{array}{ll}
# \boldsymbol{h}^{\top} \boldsymbol{s}_{t-1} & \text { dot } \\
# \boldsymbol{h}^{\top} \boldsymbol{W}_{\boldsymbol{a}} \boldsymbol{s}_{t-1} & \text { general } \\
# \boldsymbol{v}_{a}^{\top} \tanh \left(\boldsymbol{W}_{\boldsymbol{a}}\left[\boldsymbol{h} ; \boldsymbol{s}_{t-1}\right]\right) & \text { concat }
# \end{array}\right.$$
# --------

# 
# **_We wil use "concat attention"_**:
# 
# First, we calculate the *energy* between the previous decoder hidden state $s_{t-1}$ and the encoder hidden states $H$. As our encoder hidden states $H$ are a sequence of $T$ tensors, and our previous decoder hidden state $s_{t-1}$ is a single tensor, the first thing we do is `repeat` the previous decoder hidden state $T$ times. $\Rightarrow$<br>
# We have:<br>
# $$H = \bigl[\boldsymbol{h}_{0}, ..., \boldsymbol{h}_{T-1}\bigr] \\ \bigl[\boldsymbol{s}_{t-1}, ..., \boldsymbol{s}_{t-1}\bigr]$$
# 
# The encoder hidden dim and the decoder hidden dim should be equal: **dec hid dim = enc hid dim**.<br>
#  We then calculate the energy, $E_t$, between them by concatenating them together:<br>
# 
# $$\bigl[[\boldsymbol{h}_{0}, \boldsymbol{s}_{t-1}], ..., [\boldsymbol{h}_{T-1}, \boldsymbol{s}_{t-1}]\bigr]$$
# 
# And passing them through a linear layer (`attn` = $\boldsymbol{W}_{\boldsymbol{a}}$) and a $\tanh$ activation function:
# 
# $$E_t = \tanh(\text{attn}(H, s_{t-1}))$$ 
# 
# This can be thought of as calculating how well each encoder hidden state "matches" the previous decoder hidden state.
# 
# We currently have a **[enc hid dim, src sent len]** tensor for each example in the batch. We want this to be **[src sent len]** for each example in the batch as the attention should be over the length of the source sentence. This is achieved by multiplying the `energy` by a **[1, enc hid dim]** tensor, $v$.
# 
# $$\hat{a}_t = v E_t$$
# 

# We can think of this as calculating a weighted sum of the "match" over all `enc_hid_dem` elements for each encoder hidden state, where the weights are learned (as we learn the parameters of $v$).
# 
# Finally, we ensure the attention vector fits the constraints of having all elements between 0 and 1 and the vector summing to 1 by passing it through a $\text{softmax}$ layer.
# 
# $$a_t = \text{softmax}(\hat{a_t})$$
# 
# ### Temperature SoftMax
# 
# <img src="https://miro.medium.com/max/793/1*S5X1pBq_jfDreJOs7yP-ZQ.png" height=100>
# 
# This gives us the attention over the source sentence!
# 
# Graphically, this looks something like below. $z = s_{t-1}$. The green/yellow blocks represent the hidden states from both the forward and backward RNNs, and the attention computation is all done within the pink block.
# 

# ![hw_1.png](data:image/png;base64,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)

# # Neural Machine Translation
# 
# Write down some summary on your experiments and illustrate it with convergence plots/metrics and your thoughts. Just like you would approach a real problem.

# In[20]:


get_ipython().system(' wget https://drive.google.com/uc?id=1NWYqJgeG_4883LINdEjKUr6nLQPY6Yb_ -O data.txt')

# Thanks to YSDA NLP course team for the data
# (who thanks tilda and deephack teams for the data in their turn)


# In[21]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchtext
try:
  from torchtext.data import Field, BucketIterator
except:
  from torchtext.legacy.data import Field, BucketIterator

import spacy

import random
import math
import time
import numpy as np

import matplotlib
matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output

from nltk.tokenize import WordPunctTokenizer

import pandas as pd
from sklearn.model_selection import ParameterGrid

from nltk.translate.bleu_score import corpus_bleu
import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import random


# In[22]:


get_ipython().run_cell_magic('capture', '', '!pip install transformers')


# In[23]:


from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


# We'll set the random seeds for deterministic results.

# In[24]:


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# ## Preparing Data
# 
# Here comes the preprocessing

# In[25]:


tokenizer_W = WordPunctTokenizer()

def tokenize_ru(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())[::-1]

def tokenize_en(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())


# In[26]:


SRC = Field(tokenize=tokenize_ru,
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TRG = Field(tokenize=tokenize_en,
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)


dataset = torchtext.legacy.data.TabularDataset(
    path='data.txt',
    format='tsv',
    fields=[('trg', TRG), ('src', SRC)]
)


# In[27]:


print(len(dataset.examples))
print(dataset.examples[0].src)
print(dataset.examples[0].trg)


# In[28]:


train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")


# In[29]:


SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)


# In[30]:


print(f"Unique tokens in source (ru) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")


# And here is example from train dataset:

# In[31]:


print(vars(train_data.examples[9]))


# When we get a batch of examples using an iterator we need to make sure that all of the source sentences are padded to the same length, the same with the target sentences. Luckily, TorchText iterators handle this for us! 
# 
# We use a `BucketIterator` instead of the standard `Iterator` as it creates batches in such a way that it minimizes the amount of padding in both the source and target sentences. 

# In[32]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[33]:


def _len_sort_key(x):
    return len(x.src)

BATCH_SIZE = 32

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device,
    sort_key=_len_sort_key
)


# ## Let's use modules.py

# In[34]:


get_ipython().run_cell_magic('capture', '', '\n!wget https://raw.githubusercontent.com/imvladikon/deep_learning_school/master/semester_2/5_seq2seq_attention/modules.py modules.py')


# In[35]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[36]:


# !ls your_path_to_modules.py


# In[37]:


# %cd ./drive/MyDrive/your_path_to_modules.py


# ## Encoder
# 
# For a multi-layer RNN, the input sentence, $X$, goes into the first (bottom) layer of the RNN and hidden states, $H=\{h_1, h_2, ..., h_T\}$, output by this layer are used as inputs to the RNN in the layer above. Thus, representing each layer with a superscript, the hidden states in the first layer are given by:
# 
# $$h_t^1 = \text{EncoderRNN}^1(x_t, h_{t-1}^1)$$
# 
# The hidden states in the second layer are given by:
# 
# $$h_t^2 = \text{EncoderRNN}^2(h_t^1, h_{t-1}^2)$$
# 
# Extending our multi-layer equations to LSTMs, we get:
# 
# $$\begin{align*}
# (h_t^1, c_t^1) &= \text{EncoderLSTM}^1(x_t, (h_{t-1}^1, c_{t-1}^1))\\
# (h_t^2, c_t^2) &= \text{EncoderLSTM}^2(h_t^1, (h_{t-1}^2, c_{t-1}^2))
# \end{align*}$$
# 
# <br><br>
# <img src="https://drive.google.com/uc?id=1uIUxtZU8NvGdz0J9BlRSTbsBLFh32rxx">

# In[73]:


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


# ## Attention
# 
# $$\operatorname{score}\left(\boldsymbol{h}, \boldsymbol{s}_{t-1}\right)=
# \boldsymbol{v}_{a}^{\top} \tanh \left(\boldsymbol{W}_{\boldsymbol{a}}\left[\boldsymbol{h} ; \boldsymbol{s}_{t-1}\right]\right) \text { - concat attention}$$

# In[54]:


def softmax(x, dim=None, temperature = 1.):
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=dim)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, temperature = 1.):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        self.temperature = temperature

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        # return F.softmax(attention, dim=1)
        return softmax(attention, dim=0, temperature = self.temperature)


# ## Decoder with Attention
# 
# To make it really work you should also change the`Decoder` class from the classwork in order to make it to use `Attention`. 
# You may just copy-paste `Decoder` class and add several lines of code to it.
# 
# The decoder contains the attention layer `attention`, which takes the previous hidden state $s_{t-1}$, all of the encoder hidden states $H$, and returns the attention vector $a_t$.
# 
# We then use this attention vector to create a weighted source vector, $w_t$, denoted by `weighted`, which is a weighted sum of the encoder hidden states, $H$, using $a_t$ as the weights.
# 
# $$w_t = a_t H$$
# 
# The input word that has been embedded $y_t$, the weighted source vector $w_t$, and the previous decoder hidden state $s_{t-1}$, are then all passed into the decoder RNN, with $y_t$ and $w_t$ being concatenated together.
# 
# $$s_t = \text{DecoderGRU}([y_t, w_t], s_{t-1})$$
# 
# We then pass $y_t$, $w_t$ and $s_t$ through the linear layer, $f$, to make a prediction of the next word in the target sentence, $\hat{y}_{t+1}$. This is done by concatenating them all together.
# 
# $$\hat{y}_{t+1} = f(y_t, w_t, s_t)$$
# 
# The image below shows decoding the **first** word in an example translation.
# 
# The green/yellow blocks show the forward/backward encoder RNNs which output $H$, the red block is $z = s_{t-1} = s_0$, the blue block shows the decoder RNN which outputs $s_t = s_1$, the purple block shows the linear layer, $f$, which outputs $\hat{y}_{t+1}$ and the orange block shows the calculation of the weighted sum over $H$ by $a_t$ and outputs $w_t$. Not shown is the calculation of $a_t$.

# ![hw_2.png](data:image/png;base64,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It5xRgo1GpztFNnso/a1Q3yM9N1p1momDQ1P1b4cBt5oP+mJ1yg0FXG2l/Z/aXyMTC5PrisAwONKAE/3ywxNpGjo3Qw0nmaow9jew3juDJMNI1tBg7+bzrDnuY7vfHYe3H4PXbkYeO9T893hQg36wr6dNjFncHs/J2p3CM6X939MYoJMJrO9nztLe+gYOMXU66uEVmFxdxXBFQXfIIVMJ317Ron5gX5G33yLps55igdE7YoO4hscJ9MZ2jWFzCAQ6WNocpJ4tkD/iSy5rKWH14v8BRvgxd/ixe8LQqjCA8cCj2Y9JA13vU3M5gRjkPRYlffMLh6y/13H4QkSH8uTT7XtORTfaHfiCvXTnXnA1DmTtodQ2r24uQNfLMfpgXa697R1BTjme4mJdy8R+a3EwgGRN3cME8uNkoq6qS+1AUFC8RTZyUYS2QL5TDdDA9Zo99Q/MCKWVb+07trnn3L1+i/MTF3m0liC88m/XSjyL6wDJj989hE/3a4wM3WZD04anH66r0lEROQ/9iKUsEWOqBVgmrm7dubvpRnKOTkee94xydG1DPzMrVk3i8sxhl+x03r05maIiIgFKIEROfTWgDJz129wc/omcxUw/4T6VLclFsxW/IUM+YibiLX7auRQMIEyt65e49eZWWYrsFEDqABLFO0JQqdyjISbaHkBhoiJiMjhowRG5NCrAsvM3bjClS++4vslKFehfrFgC+Mffswb71wk72ffrb0i/5wJLHH7x6/55stv+W4JzA1g88anM59c4NW3z3LChcXPoYiIiFXpDIzIobcBrLH6+yIr90uU17cq4o2AHW9LBCMUxO/YGsUo8izqz9vKvQVKpTLl6tblto1AE77IcYyAgeHYGfggIiLyf1ICIyIiIiIilqF6rYiIiIiIWIYSGBERERERsQwlMCIiIiIiYhlKYERERERExDKUwIiIiIiIiGUogREREREREctQAiMiIiIiIpahBEZERERERCxDCYyIiIiIiFiGEhgREREREbGMJxBAppaCaopfAAAAAElFTkSuQmCC)

# ### Teacher forcing
# 
# Teacher forcing is a method for quickly and efficiently training recurrent neural network models that use the ground truth from a prior time step as input.

# <img src="https://miro.medium.com/max/421/1*U3d8D_GnfW13Y3nDgvwJSw.png">
# 
# When training/testing our model, we always know how many words are in our target sentence, so we stop generating words once we hit that many. During inference (i.e. real world usage) it is common to keep generating words until the model outputs an `<eos>` token or after a certain amount of words have been generated.
# 
# Once we have our predicted target sentence, $\hat{Y} = \{ \hat{y}_1, \hat{y}_2, ..., \hat{y}_T \}$, we compare it against our actual target sentence, $Y = \{ y_1, y_2, ..., y_T \}$, to calculate our loss. We then use this loss to update all of the parameters in our model.
# 
# 

# In[74]:


class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), a


# ## Seq2Seq
# 
# Main idea:
# * $w_t = a_t H$
# 
# * $s_t = \text{DecoderGRU}([y_t, w_t], s_{t-1})$
# 
# * $\hat{y}_{t+1} = f(y_t, w_t, s_t)$
# 
# **Note**: our decoder loop starts at 1, not 0. This means the 0th element of our `outputs` tensor remains all zeros. So our `trg` and `outputs` look something like:
# 
# $$\begin{align*}
# \text{trg} = [<sos>, &y_1, y_2, y_3, <eos>]\\
# \text{outputs} = [0, &\hat{y}_1, \hat{y}_2, \hat{y}_3, <eos>]
# \end{align*}$$
# 
# Later on when we calculate the loss, we cut off the first element of each tensor to get:
# 
# $$\begin{align*}
# \text{trg} = [&y_1, y_2, y_3, <eos>]\\
# \text{outputs} = [&\hat{y}_1, \hat{y}_2, \hat{y}_3, <eos>]
# \end{align*}$$

# In[69]:


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        inp = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, attn_weights = self.decoder(inp, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            inp = trg[t] if teacher_force else top1
        return outputs


# ## Training

# In[40]:


#we use code from notebook

# For reloading 
# from modules import *
# import modules
# import imp
# imp.reload(modules)

# Encoder = modules.Encoder
# Attention = modules.Attention
# Decoder = modules.DecoderWithAttention
# Seq2Seq = modules.Seq2Seq


# In[41]:


def train(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None, scheduler=None):
    model.train()
    
    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        
        output = output[1:].view(-1, OUTPUT_DIM)
        trg = trg[1:].view(-1)
        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        # Let's clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        if scheduler is not None:
          scheduler.step() 
        
        epoch_loss += loss.item()
        
        history.append(loss.cpu().data.numpy())
        if (i+1)%10==0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            clear_output(True)
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
                ax[1].set_title(history[-1])
            if valid_history is not None:
                ax[1].plot(valid_history, label='general valid history')
            plt.legend()
            
            plt.show()

        
    return epoch_loss / len(iterator)

@torch.no_grad()
def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    history = []
    
    for i, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        output = model(src, trg, 0) #turn off teacher forcing

        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, OUTPUT_DIM)
        trg = trg[1:].view(-1)

        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[42]:


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BIDIRECTIONAL = True


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attention = Attention(ENC_HID_DIM, DEC_HID_DIM)
dec = DecoderWithAttention(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attention)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)
        
model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


train_history = []
valid_history = []

N_EPOCHS = 12
CLIP = 5
PAD_IDX = TRG.vocab.stoi['<pad>']
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP, train_history, valid_history)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best-val-model.pt')
    
    train_history.append(train_loss)
    valid_history.append(valid_loss)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


# __Let's take a look at our network quality__:

# In[43]:


def cut_on_eos(tokens_iter):
    for token in tokens_iter:
        if token == '<eos>':
            break
        yield token

def remove_tech_tokens(tokens_iter, tokens_to_remove=['<sos>', '<unk>', '<pad>']):
    return [x for x in tokens_iter if x not in tokens_to_remove]

def generate_translation(src, trg, model, TRG_vocab):
    model.eval()

    output = model(src, trg, 0) #turn off teacher forcing
    output = output[1:].argmax(-1)

    original = remove_tech_tokens(cut_on_eos([TRG_vocab.itos[x] for x in list(trg[:,0].cpu().numpy())]))
    generated = remove_tech_tokens(cut_on_eos([TRG_vocab.itos[x] for x in list(output[:, 0].cpu().numpy())]))
    
    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()

def get_text(x, TRG_vocab):
     generated = remove_tech_tokens(cut_on_eos([TRG_vocab.itos[elem] for elem in list(x)]))
     return generated


# In[44]:


#model.load_state_dict(torch.load('best-val-model.pt'))
batch = next(iter(test_iterator))

for idx in range(10):
    src = batch.src[:, idx:idx+1]
    trg = batch.trg[:, idx:idx+1]
    generate_translation(src, trg, model, TRG.vocab)


# ## Bleu
# 
# [link](https://www.aclweb.org/anthology/P02-1040.pdf)
# 
# <img src="https://drive.google.com/uc?id=1umJF2S8PiayxD9Xo8xvjW8QsrSLidozD" height=400>

# вообще судя по повторяющимся предсказаниям, наши рнн-ки испытывают проблемки при обучении и неплохо бы изменить clip значение, поменять может дропаут (а может вообще varitional dropout) но это мы оставим за скобками ноутбука.

# In[45]:


original_text = []
generated_text = []
model.eval()
with torch.no_grad():

    for i, batch in tqdm.tqdm(enumerate(test_iterator)):

        src = batch.src
        trg = batch.trg

        output = model(src, trg, 0) #turn off teacher forcing

        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]

        output = output[1:].argmax(-1)
        
        original_text.extend([get_text(x, TRG.vocab) for x in trg.cpu().numpy().T])
        generated_text.extend([get_text(x, TRG.vocab) for x in output.detach().cpu().numpy().T])

# original_text = flatten(original_text)
# generated_text = flatten(generated_text)


# In[46]:


corpus_bleu([[text] for text in original_text], generated_text) * 100


# был получен довольно хороший результат

# ## Recommendations:
# * use bidirectional RNN
# * change learning rate from epoch to epoch
# * when classifying the word don't forget about embedding and summa of encoders state 
# * you can use more than one layer

# ## You will get:
# 
# * `2` points if `21 < bleu score < 23`
# * `4` points if `23 < bleu score < 25`
# * `7` points if `25 < bleu score < 27`
# * `9` points if `27 < bleu score < 29`
# * `10` points if `bleu score > 29`
# 
# When your result is checked, your 10 translations will be checked too
# 

# ## Your Conclusion
# * information about your the results obtained 
# * difference between seminar and homework model

# макисмальный результат = 31 в целом нужно увеличивать аккуратно hid_dim при определенном dropout. возможно стоит также поиграться с VariationalDropout (имплементация например https://github.com/yeahrmek/rnn_vd). также нужно подбирать соответствующую температуру для сглаживания

# ## References
# 
# * https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html