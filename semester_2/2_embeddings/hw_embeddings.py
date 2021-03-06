%%capture
%%bash

gdown https://drive.google.com/uc?id=1eE1FiUkXkcbw0McId4i7qY-L8hH-_Qph
unzip archive.zip

%%capture
!pip install datashader
!pip install umap-learn
!pip install transformers

import math
import random
import string

import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
import time
import os
import copy


import nltk
import gensim
import gensim.downloader as api

from sklearn.decomposition import PCA
from typing import Dict 
from sklearn.feature_extraction.text import TfidfVectorizer

import bokeh.models as bm, bokeh.plotting as pl
from bokeh.io import output_notebook
output_notebook()
import umap
import umap.plot


from __future__ import print_function, division
import gc
from IPython.display import clear_output
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from torch.nn import functional as F
from tqdm.notebook import tqdm

def clear_cache():
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'png'
# the next line provides graphs of better quality on HiDPI screens
%config InlineBackend.figure_format = 'retina'
plt.style.use('seaborn')
%config InlineBackend.print_figure_kwargs={'facecolor' : "w"}
plt.ion()   # interactive mode

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)
torch.cuda.random.manual_seed(42)
torch.cuda.random.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin", header=None, names=["emotion", "id", "date", "flag", "user", "text"])

data.head()

data.emotion.value_counts()

data.emotion = data.emotion.map({0:0, 4:1})

examples = data["text"].sample(10)
print("\n".join(examples))

indexes = np.arange(data.shape[0])
np.random.shuffle(indexes)
dev_size = math.ceil(data.shape[0] * 0.8)

dev_indexes = indexes[:dev_size]
test_indexes = indexes[dev_size:]

dev_data = data.iloc[dev_indexes]
test_data = data.iloc[test_indexes]

dev_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

tokenizer = nltk.WordPunctTokenizer()
line = tokenizer.tokenize(dev_data["text"][0].lower())
print(" ".join(line))

filtered_line = [w for w in line if all(c not in string.punctuation for c in w) and len(w) > 3]
print(" ".join(filtered_line))

word2vec = api.load("word2vec-google-news-300")

emb_line = [word2vec.get_vector(w) for w in filtered_line if w in word2vec]
print(sum(emb_line).shape)

mean = np.mean(word2vec.vectors, 0)
std = np.std(word2vec.vectors, 0)
norm_emb_line = [(word2vec.get_vector(w) - mean) / std for w in filtered_line if w in word2vec and len(w) > 3]
print(sum(norm_emb_line).shape)
print([all(norm_emb_line[i] == emb_line[i]) for i in range(len(emb_line))])

class TwitterDataset(Dataset):
    def __init__(self, data: pd.DataFrame, feature_column: str, target_column: str, word2vec: gensim.models.Word2Vec):
        self.tokenizer = nltk.WordPunctTokenizer()
        
        self.data = data

        self.feature_column = feature_column
        self.target_column = target_column

        self.word2vec = word2vec

        self.label2num = lambda label: 0 if label == 0 else 1
        self.mean = np.mean(word2vec.vectors, axis=0)
        self.std = np.std(word2vec.vectors, axis=0)

    def __getitem__(self, item):
        text = self.data[self.feature_column][item]
        label = self.label2num(self.data[self.target_column][item])

        tokens = self.get_tokens_(text)
        embeddings = self.get_embeddings_(tokens)

        return {"features": embeddings, "targets": label}

    def get_tokens_(self, text):
        return [w for w in self.tokenizer.tokenize(text) if not any(c in string.punctuation for c in w) and len(w) > 3]

    def get_embeddings_(self, tokens):
        embeddings = [(self.word2vec.get_vector(token) - self.mean) / self.std for token in tokens if token in word2vec and len(token) > 3]

        if len(embeddings) == 0:
            embeddings = np.zeros((1, self.word2vec.vector_size))
        else:
            embeddings = np.array(embeddings)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(-1, 1)

        return embeddings

    def __len__(self):
        return self.data.shape[0]

dev = TwitterDataset(dev_data, "text", "emotion", word2vec)

indexes = np.arange(len(dev))
np.random.shuffle(indexes)
example_indexes = indexes[::1000]

examples = {"features": [np.sum(dev[i]["features"], axis=0) for i in example_indexes], 
            "targets": [dev[i]["targets"] for i in example_indexes]}
print(len(examples["features"]))

pca = PCA(n_components=2)
examples["transformed_features"] = pca.fit_transform(examples["features"])

def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    """ draws an interactive plot for data points with auxilirary info on hover """
    data_source = bm.ColumnDataSource({ 'x' : x, 'y' : y, 'color': color, **kwargs })

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show: pl.show(fig)
    return fig

draw_vectors(
    examples["transformed_features"][:, 0], 
    examples["transformed_features"][:, 1], 
    color=[["red", "blue"][t] for t in examples["targets"]]
    )

embeddings=umap.UMAP(n_neighbors=100,min_dist=1,metric='correlation').fit(np.array(examples["features"]))

umap.plot.points(embeddings, labels=np.array(examples["targets"]), color_key_cmap='Paired', background='black')

try:
  del embeddings
except:
  pass

batch_size = 1024
num_workers = 4

def average_emb(batch):
    features = [np.mean(b["features"], axis=0) for b in batch]
    targets = [b["targets"] for b in batch]

    return {"features": torch.FloatTensor(features), "targets": torch.LongTensor(targets)}


train_size = math.ceil(len(dev) * 0.8)

train, valid = random_split(dev, [train_size, len(dev) - train_size])

train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, collate_fn=average_emb)
valid_loader = DataLoader(valid, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=average_emb)

def accuracy(outputs, labels):
    predicted = torch.argmax(outputs, dim=1)
    return torch.mean(torch.eq(predicted, labels).float())

def training(model, optimizer, criterion, train_loader, epoch, device="cpu"):
    pbar = tqdm(train_loader, desc=f"Epoch {e + 1}. Train Loss: {0}")
    model.train()
    for batch in pbar:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)

        model.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        if scheduler is not None: scheduler.step()
        optimizer.zero_grad()

        acc = accuracy(outputs, targets).item()

        pbar.set_description(f"Epoch {e + 1}. Train Loss: {loss:.4} Train Acc:{acc:.4}")
    

def testing(model, criterion, test_loader, device="cpu"):
    pbar = tqdm(test_loader, desc=f"Test Loss: {0}, Test Acc: {0}")
    mean_loss = 0
    mean_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in pbar:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)
            acc = accuracy(outputs, targets)

            mean_loss += loss.item()
            mean_acc += acc.item()

            pbar.set_description(f"Test Loss: {loss:.4}, Test Acc: {acc:.4}")

    pbar.set_description(f"Test Loss: {mean_loss / len(test_loader):.4}, Test Acc: {mean_acc / len(test_loader):.4}")

    return {"Test Loss": mean_loss / len(test_loader), "Test Acc": mean_acc / len(test_loader)}

class LinearModel(nn.Module):

    def __init__(self, vector_size, num_classes):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(vector_size, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        
        self.fc2 = nn.Linear(1024, 128)
        self.bn2 = nn.BatchNorm1d(num_features=128)

        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, num_classes)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=mean.mean(), std=std.mean())
                m.bias.data.zero_()

    def forward(self, X):
        out = F.relu(self.bn1(self.fc1(X)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.fc3(out))
        out = F.softmax(self.fc4(out), dim=1)
        return out

# Не забудь поиграться с параметрами ;)
vector_size = dev.word2vec.vector_size
num_classes = 2
lr = 1e-2
num_epochs = 1

model = LinearModel(vector_size=vector_size, num_classes=num_classes)
model = model.to(device)
criterion =  nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
total_steps = len(train_loader) * num_epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps)

best_metric = np.inf
for e in range(num_epochs):
    training(model, optimizer, criterion, train_loader, e, device)
    log = testing(model, criterion, valid_loader, device)
    print(log)
    if log["Test Loss"] < best_metric:
        torch.save(model.state_dict(), "model.pt")
        best_metric = log["Test Loss"]

import copy

class AbstractModel(nn.Module):
  def __init__(self, cls2idx, *args, **kwargs):
    super(AbstractModel, self).__init__()
    self.metric_history = {"train":[], "val":[]} 
    self.cls2idx = cls2idx
    self.num_classes= len(cls2idx)
    self.idx2cls = {v:k for k,v in self.cls2idx.items()}
    
  @torch.no_grad()
  def predict_sample(self, sample, return_class=False):
    self.eval()
    outputs = self(sample) 
    if return_class:
      _, preds = torch.max(outputs, 1) 
      return self.idx2cls[preds.item()]
    else:
      return outputs

  def __freeze__(self):
    for layer in list(self.children()):
      for param in layer.parameters():
        param.requires_grad = False
  
  def init_weights(self, modules=None):
    for m in modules:
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

  
  @torch.no_grad()
  def testing(self, criterion, test_loader):
    model = self
    pbar = tqdm(test_loader, desc=f"Test Loss: {0}, Test Acc: {0}")
    mean_loss = 0
    mean_acc = 0
    device = model.device
    model.eval()
    for batch in pbar:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(features)
        loss = criterion(outputs, targets)
        acc = accuracy(outputs, targets)

        mean_loss += loss.item()
        mean_acc += acc.item()

        pbar.set_description(f"Test Loss: {loss:.4}, Test Acc: {acc:.4}")

    pbar.set_description(f"Test Loss: {mean_loss / len(test_loader):.4}, Test Acc: {mean_acc / len(test_loader):.4}")

    return {"Test Loss": mean_loss / len(test_loader), "Test Acc": mean_acc / len(test_loader)}

    
  def fit(self, dataloaders, criterion, optimizer, scheduler, num_epochs=25, do_plot=True):
    best_metric = np.inf
    self.metric_history = {"train":[], "val":[]} 
    model = self
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes = {"train": len(dataloaders["train"]),  'val': len(dataloaders["val"])}
    for epoch in range(num_epochs):
        clear_cache()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   
            running_loss = 0.0
            running_corrects = 0
            for batch in dataloaders[phase]:
                inputs = batch["features"].to(device)
                labels = batch["targets"].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs) 
                    _, preds = torch.max(outputs, 1) 
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward() 
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                running_loss += loss.item()
                acc = accuracy(outputs, labels)
                running_corrects += acc
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                  scheduler.step(epoch_loss)
                else:
                  scheduler.step()  
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            self.metric_history[phase].append({"epoch": epoch,"loss":epoch_loss, "acc":epoch_acc.item()})
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
        if do_plot:
          self.plot_learning_curves(title='Epoch {}/{}'.format(epoch, num_epochs - 1))
        
    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    if do_plot:
      self.plot_learning_curves()
    return self.metric_history


  def plot_learning_curves(self, do_clear=True, title=""):
    if do_clear:
      clear_output(wait=True)
    with plt.style.context('seaborn-whitegrid'):
      fig,ax = plt.subplots(1,2, figsize=(16, 6))
      train_history = pd.DataFrame(self.metric_history["train"])
      val_history = pd.DataFrame(self.metric_history["val"])
      train_history.plot(x="epoch", y="acc", ax=ax[0], color="r", label="acc_train") 
      val_history.plot(x="epoch", y="acc", ax=ax[0], color="b", label="acc_val")
      train_history.plot(x="epoch", y="loss", color="r", ax=ax[1], label="loss_train")
      val_history.plot(x="epoch", y="loss", color="b", ax=ax[1], label="loss_val")
      ax[0].set_title(f'Train Acc: {train_history.iloc[-1]["acc"]:.4f} Val Acc: {val_history.iloc[-1]["acc"]:.4f}')
      ax[1].set_title(f'Train Loss: {train_history.iloc[-1]["loss"]:.4f} Val Loss: {val_history.iloc[-1]["loss"]:.4f}')
      if not title:
        fig.suptitle(title)
      plt.show();

class LinearModel(AbstractModel):

    def __init__(self, cls2idx, vector_size):
        super(LinearModel, self).__init__(cls2idx)
        self.fc1 = nn.Linear(vector_size, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        
        self.fc2 = nn.Linear(1024, 128)
        self.bn2 = nn.BatchNorm1d(num_features=128)

        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, self.num_classes)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=mean.mean(), std=std.mean())
                m.bias.data.zero_()

    def forward(self, X):
        out = F.relu(self.bn1(self.fc1(X)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.fc3(out))
        out = F.softmax(self.fc4(out), dim=1)
        return out

dev = TwitterDataset(dev_data, "text", "emotion", word2vec)

batch_size = 1024
num_workers = 4

def average_emb(batch):
    features = [np.mean(b["features"], axis=0) for b in batch]
    targets = [b["targets"] for b in batch]

    return {"features": torch.FloatTensor(features), "targets": torch.LongTensor(targets)}


train_size = math.ceil(len(dev) * 0.8)
train, valid = random_split(dev, [train_size, len(dev) - train_size])
train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, collate_fn=average_emb)
valid_loader = DataLoader(valid, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=average_emb)

dataloaders = {
    'train':train_loader,
    'val':valid_loader
  }

cls2idx = {"0":0,"1":1}
num_classes = len(cls2idx)
lr = 1e-2
num_epochs = 5

model = LinearModel(cls2idx, vector_size=dev.word2vec.vector_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps)

history = model.fit(dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs)

test_loader = DataLoader(
    TwitterDataset(test_data, "text", "emotion", word2vec), 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=False,
    drop_last=False, 
    collate_fn=average_emb)

# у нас в fit, загружается лучшая модель, так что эта строка лишняя
# model.load_state_dict(torch.load("model.pt", map_location=device))

print(testing(model, criterion, test_loader, device=device))

class ClassifierLSTM(AbstractModel):
    
    def __init__(self, cls2idx, embedding_dim, n_hidden, n_output, n_layers, dropout = 0.1):
        super(ClassifierLSTM, self).__init__(cls2idx)
        
        self.n_layers = n_layers   
        self.n_hidden = n_hidden   
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(self.embedding_dim, n_hidden, n_layers, batch_first = True, dropout = dropout)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(n_hidden, n_output),
                                         nn.Softmax() )
        
        
    def forward (self, inputs):
        out, _ = self.lstm(inputs.unsqueeze(1), None)
        out = out[:, -1, :] 
        out = self.classifier(out)
        return out

lr = 1e-2

model = ClassifierLSTM(cls2idx, dev.word2vec.vector_size, n_hidden=512, n_output=2, n_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps)


num_epochs = 5
history = model.fit(dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs)

test_loader = DataLoader(
    TwitterDataset(test_data, "text", "emotion", word2vec), 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=False,
    drop_last=False, 
    collate_fn=average_emb)

print(testing(model, criterion, test_loader, device=device))

class TwitterDatasetOOV(TwitterDataset):
    def __init__(self, data: pd.DataFrame, feature_column: str, target_column: str, word2vec: gensim.models.Word2Vec):
        super(TwitterDatasetOOV, self).__init__(data, feature_column, target_column, word2vec)

    def get_embeddings_(self, tokens):
        embeddings = np.zeros((len(tokens), self.word2vec.vector_size))
        oov_indices = []
        for idx, token in enumerate(tokens):
          if token in self.word2vec:
            embeddings[idx] = (self.word2vec.get_vector(token) - self.mean) / self.std
          else:
            oov_indices.append(idx)
        if len(oov_indices)>0:
          embeddings[oov_indices] = np.nansum(embeddings, axis=0)
        if len(embeddings) == 0:
            embeddings = np.zeros((1, self.word2vec.vector_size))            
        else:
            embeddings = np.array(embeddings)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(-1, 1)
        return embeddings

dev = TwitterDatasetOOV(dev_data, "text", "emotion", word2vec)
batch_size = 1024
num_workers = 4
def average_emb(batch):
    features = [np.mean(b["features"], axis=0) for b in batch]
    targets = [b["targets"] for b in batch]
    return {"features": torch.FloatTensor(features), "targets": torch.LongTensor(targets)}
train_size = math.ceil(len(dev) * 0.8)
train, valid = random_split(dev, [train_size, len(dev) - train_size])
train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, collate_fn=average_emb)
valid_loader = DataLoader(valid, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=average_emb)

dataloaders = {
    'train':train_loader,
    'val':valid_loader
  }
lr = 1e-2
model = LinearModel(cls2idx, vector_size=dev.word2vec.vector_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps)

num_epochs = 5
history = model.fit(dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs)

class TwitterDatasetTfIdf(TwitterDataset):
    def __init__(self, data: pd.DataFrame, feature_column: str, target_column: str, word2vec: gensim.models.Word2Vec, weights: Dict[str, float] = None):
        super().__init__(data, feature_column, target_column, word2vec)

        if weights is None:
            self.weights = self.get_tf_idf_()
        else:
            self.weights = weights

    def get_embeddings_(self, tokens):
        embeddings = [(self.word2vec.get_vector(token) - self.mean) / self.std  * self.weights.get(token, 1) for token in tokens if token in word2vec and len(token) > 3]

        if len(embeddings) == 0:
            embeddings = np.zeros((1, self.word2vec.vector_size))
        else:
            embeddings = np.array(embeddings)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(-1, 1)

        return embeddings

    def get_tf_idf_(self):
        tokenized_texts = self.data["text"].tolist()
        tf_idf = TfidfVectorizer()
        tf_idf.fit(tokenized_texts)
        return dict(zip(tf_idf.get_feature_names(), tf_idf.idf_))


dev = TwitterDatasetTfIdf(dev_data, "text", "emotion", word2vec)

batch_size = 1024
num_workers = 4

def average_emb(batch):
    features = [np.mean(b["features"], axis=0) for b in batch]
    targets = [b["targets"] for b in batch]

    return {"features": torch.FloatTensor(features), "targets": torch.LongTensor(targets)}


train_size = math.ceil(len(dev) * 0.8)

train, valid = random_split(dev, [train_size, len(dev) - train_size])

train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, collate_fn=average_emb)
valid_loader = DataLoader(valid, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=average_emb)

dataloaders = {
    'train':train_loader,
    'val':valid_loader
  }

  
vector_size = dev.word2vec.vector_size
lr = 1e-2
num_epochs = 5

model = LinearModel(cls2idx, vector_size=vector_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps)


history = model.fit(dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs)

%%capture

!pip install fasttext
!wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
!gunzip cc.en.300.bin.gz

import fasttext as ft

ft_model = ft.load_model("cc.en.300.bin")

from torch.utils.data import Dataset, random_split


class TwitterDatasetFasttext(Dataset):
    def __init__(self, data: pd.DataFrame, feature_column: str, target_column: str, word2vec: gensim.models.Word2Vec, fasttext: ft.FastText._FastText):
        self.tokenizer = nltk.WordPunctTokenizer()
        
        self.data = data

        self.feature_column = feature_column
        self.target_column = target_column

        self.word2vec = word2vec
        self.fasttext = fasttext

        self.label2num = lambda label: 0 if label == 0 else 1
        self.mean = np.mean(word2vec.vectors, axis=0)
        self.std = np.std(word2vec.vectors, axis=0)

        assert fasttext.get_dimension() == word2vec.vector_size

    def __getitem__(self, item):
        text = self.data[self.feature_column][item]
        label = self.label2num(self.data[self.target_column][item])

        tokens = self.get_tokens_(text)
        embeddings = self.get_embeddings_(tokens)

        return {"features": embeddings, "targets": label}

    def get_tokens_(self, text):
        return [w for w in self.tokenizer.tokenize(text) if not any(c in string.punctuation for c in w) and len(w) > 3]

    def get_embeddings_(self, tokens):
        embeddings = np.zeros((len(tokens), self.word2vec.vector_size))
        oov_indices = []
        for idx, token in enumerate(tokens):
          if token in self.word2vec:
            embeddings[idx] = (self.word2vec.get_vector(token) - self.mean) / self.std
          else:
            oov_indices.append(idx)
        if len(oov_indices)>0:
          embeddings[oov_indices] = np.nansum(embeddings, axis=0)
        
        if len(embeddings) == 0:
            embeddings = np.zeros((1, self.word2vec.vector_size))            
        else:
            embeddings = np.array(embeddings)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(-1, 1)
        embeddings = embeddings + ft_model.get_sentence_vector(" ".join(tokens))
        return embeddings

    def __len__(self):
        return self.data.shape[0]

batch_size = 1024
num_workers = 4

dev = TwitterDatasetFasttext(dev_data, "text", "emotion", word2vec, ft_model)
def average_emb(batch):
    features = [np.mean(b["features"], axis=0) for b in batch]
    targets = [b["targets"] for b in batch]

    return {"features": torch.FloatTensor(features), "targets": torch.LongTensor(targets)}


train_size = math.ceil(len(dev) * 0.8)

train, valid = random_split(dev, [train_size, len(dev) - train_size])

train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, collate_fn=average_emb)
valid_loader = DataLoader(valid, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=average_emb)

dataloaders = {
    'train':train_loader,
    'val':valid_loader
  }

lr = 1e-2

model = LinearModel(cls2idx, vector_size=dev.word2vec.vector_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps)


num_epochs = 2
history = model.fit(dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

dataloaders = {
    'train':train_loader,
    'val':valid_loader
  }

lr = 1e-2

model = LinearModel(cls2idx, vector_size=dev.word2vec.vector_size).to(device)
criterion = LabelSmoothingCrossEntropy()
optimizer = torch.optim.Adam(model.parameters())
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps)


num_epochs = 2
history = model.fit(dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs)

data

tokenizer = nltk.WordPunctTokenizer() 

lens_t = []
lens_oov = []
for i, row in data.iterrows():
  tokens = tokenizer.tokenize(row["text"])
  lens_t.append(len(tokens))
  lens_oov.append(sum(1 for t in tokens if t not in word2vec))

data["lens_oov"] = lens_oov
data["lens_t"] = lens_t

def display_group_density_plot(df, groupby, on, palette = None, figsize = None, title="", ax=None): 
    """
    Displays a density plot by group, given a continuous variable, and a group to split the data by
    :param df: DataFrame to display data from
    :param groupby: Column name by which plots would be grouped (Categorical, maximum 10 categories)
    :param on: Column name of the different density plots
    :param palette: Color palette to use for drawing
    :param figsize: Figure size
    :return: matplotlib.axes._subplots.AxesSubplot object
    """
    if palette is None:
      palette = sns.color_palette('Set2')
    if figsize is None:
      figsize = (10, 5)
    if not isinstance(df, pd.core.frame.DataFrame):
        raise ValueError('df must be a pandas DataFrame')

    if not groupby:
        raise ValueError('groupby parameter must be provided')

    elif not groupby in df.keys():
        raise ValueError(groupby + ' column does not exist in the given DataFrame')

    if not on:
        raise ValueError('on parameter must be provided')

    elif not on in df.keys():
        raise ValueError(on + ' column does not exist in the given DataFrame')

    if len(set(df[groupby])) > 10:
        groups = df[groupby].value_counts().index[:10]

    else:
        groups = set(df[groupby])

    # Get relevant palette
    if palette:
        palette = palette[:len(groups)]
    else:
        palette = sns.color_palette()[:len(groups)]

    if ax is None:
      fig = plt.figure(figsize=figsize)
      ax = fig.add_subplot(111)
    
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    for value, color in zip(groups, palette):
        sns.kdeplot(df.loc[df[groupby] == value][on], \
                    shade=True, color=color, label=value, ax=ax)
    if not title:
      title = str("Distribution of " + on + " per " + groupby + " group")
    
    ax.set_title(title,fontsize=16)
    ax.set_xlabel(on, fontsize=16)
    return ax 

display_group_density_plot(data, groupby="emotion", on="lens_t")

display_group_density_plot(data, groupby="emotion", on="lens_oov")
