%%capture 
%%bash

gdown https://drive.google.com/uc?id=1eE1FiUkXkcbw0McId4i7qY-L8hH-_Qph&export=download

%%capture 
%%bash

pip install contractions # replacing contractions - I've > I have 
pip install git+https://github.com/NeelShah18/emot # replacing emoticons - ;) > wink
pip install optuna
pip install transformers

import contractions
import emot

import math
import random
import string
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import regex as re
import torch
import gensim
import gensim.downloader as api
from html import unescape
from collections import defaultdict
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, random_split
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from torch.nn import functional as F
torch.__version__

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english')) 

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)
torch.cuda.random.manual_seed(42)
torch.cuda.random.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

%%capture
!pip install ipython-autotime -qq
%load_ext autotime

data = pd.read_csv("archive.zip", encoding="latin", header=None, names=["emotion", "id", "date", "flag", "user", "text"])

data.head()

def fix_emoticon(s):
  """
  replacing emoticons by related meaning word
  example: ";)" - ":wink:" 
  """
  d = emot.emoticons(s)
  if not d.get("flag", False): return s
  idx=0
  r = ""
  for meaning, location in zip(d["mean"], d["location"]):
    start, end = location
    r+=s[idx:start]+":"+meaning.split()[0].lower()+":"
    idx=end
  del d
  return r

def clean_column(col:pd.Series) -> pd.Series:
  #fixing html entities
  col = np.vectorize(unescape)(col) 
  col = pd.Series(np.vectorize(contractions.fix)(col)) 
  # removing urls
  col = col.replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True) 
  col = pd.Series(np.vectorize(fix_emoticon)(col))
  #user
  col = col.str.replace('@[A-Za-z0-9]+', '') 
  col = col.str.strip()
  col = col.str.strip(string.punctuation)
  #repeated spaces
  col = col.replace(r'\s{2,}', ' ')  
  col = col.str.lower()
  col = pd.Series(np.vectorize(lambda s: " ".join(t for t in s.split() if t not in stop_words))(col))
  return col

data["text"] = clean_column(data["text"])

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
        return {"feature": embeddings, "target": label, "text": text}

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

examples = {"features": [np.sum(dev[i]["feature"], axis=0) for i in example_indexes], 
            "targets": [dev[i]["target"] for i in example_indexes]}
print(len(examples["features"]))

from sklearn.decomposition import PCA


pca = PCA(n_components=2)
examples["transformed_features"] = pca.fit_transform(examples["features"])

import bokeh.models as bm, bokeh.plotting as pl
from bokeh.io import output_notebook
output_notebook()

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

batch_size = 1024
num_workers = 4

def average_emb(batch):
    features = [np.mean(b["feature"], axis=0) for b in batch]
    targets = [b["target"] for b in batch]

    return {"features": torch.FloatTensor(features), "targets": torch.LongTensor(targets)}


train_size = math.ceil(len(dev) * 0.8)

train, valid = random_split(dev, [train_size, len(dev) - train_size])

train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, collate_fn=average_emb)
valid_loader = DataLoader(valid, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=average_emb)

def accuracy(outputs, labels):
    predicted = torch.argmax(outputs, dim=1)
    return torch.mean(torch.eq(predicted, labels).float()).item()

def training(model, optimizer, criterion, train_loader, epoch, device="cpu", scheduler=None):
    pbar = tqdm(train_loader, desc=f"Epoch {e + 1}. Train Loss: {0}")
    model.train()
    for i, batch in enumerate(pbar):
        features = batch["features"].to(device)
        targets = batch["targets"].float().unsqueeze(1).to(device)

        model.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None: scheduler.step()
        optimizer.zero_grad()

        acc = accuracy(outputs, targets)

        pbar.set_description(f"Epoch:{e + 1}.Train Loss:{loss:.4} Acc:{acc:.4}")
        
        if i%100==0:gc.collect();torch.cuda.empty_cache();
    

def testing(model, criterion, test_loader, device="cpu"):
    pbar = tqdm(test_loader, desc=f"Test Loss: {0}, Test Acc: {0}")
    mean_loss = 0
    mean_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in pbar:
            features = batch["features"].to(device)
            targets = batch["targets"].float().to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)
            acc = accuracy(outputs, targets)

            mean_loss += loss.item()
            mean_acc += acc

            pbar.set_description(f"Test Loss: {loss:.4}, Test Acc: {acc:.4}")

    pbar.set_description(f"Test Loss: {mean_loss / len(test_loader):.4}, Test Acc: {mean_acc / len(test_loader):.4}")

    return {"Test Loss": mean_loss / len(test_loader), "Test Acc": mean_acc / len(test_loader)}

class LinearModel(nn.Module):

    def __init__(self, vector_size, num_classes):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(vector_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, X):
        out = F.relu(self.fc1(X))
        out = F.relu(self.fc2(out))
        return out

vector_size = dev.word2vec.vector_size
num_classes = 1
lr = 1e-2
num_epochs = 2

model = LinearModel(vector_size=vector_size, num_classes=num_classes)
model = model.cuda()

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters())
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps)

best_metric = np.inf
for e in range(num_epochs):
    training(model, optimizer, criterion, train_loader, e, device, scheduler)
    log = testing(model, criterion, valid_loader, device)
    print(log)
    if log["Test Loss"] < best_metric:
        torch.save(model.state_dict(), "model.pt")
        best_metric = log["Test Loss"]

class ClassifierLSTM(nn.Module):
    
    def __init__(self, embedding_dim, n_hidden, n_output, n_layers, drop_p = 0.1):
        super(ClassifierLSTM, self).__init__()
        
        self.n_layers = n_layers   
        self.n_hidden = n_hidden   
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(self.embedding_dim, n_hidden, n_layers, batch_first = True, dropout = drop_p)
        self.classifier = nn.Sequential( nn.Dropout(drop_p),
                                         nn.Linear(n_hidden, 1),
                                         nn.Sigmoid() )
        
        
    def forward (self, inputs):
        out, _ = self.lstm(inputs.unsqueeze(1), None)
        out = out[:, -1, :] 
        out = self.classifier(out)
        return out

vector_size = dev.word2vec.vector_size
num_classes = 2
lr = 5e-5
num_epochs = 5

model = ClassifierLSTM(vector_size, n_hidden=512, n_output=1, n_layers=2)
model = model.cuda()
criterion = nn.BCELoss()
total_steps = len(train_loader) * num_epochs
optimizer = AdamW(model.parameters(),lr=5e-5,eps=1e-8)
        
scheduler = CosineAnnealingLR(optimizer, T_max=100)
best_metric = np.inf
for e in range(num_epochs):
    training(model, optimizer, criterion, train_loader, e, device, scheduler=scheduler)
    log = testing(model, criterion, valid_loader, device)
    print(log)
    if log["Test Loss"] < best_metric:
        torch.save(model.state_dict(), "model.pt")
        best_metric = log["Test Loss"]

test_loader = DataLoader(
    TwitterDataset(test_data, "text", "emotion", word2vec), 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=False,
    drop_last=False, 
    collate_fn=average_emb)

model.load_state_dict(torch.load("model.pt", map_location=device))

print(testing(model, criterion, test_loader, device=device))

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
        # Надо обучить tfidf на очищенном тексте. Но он принимает только список текстов, а не список списка токенов. Надо превратить второе в первое
        tokenized_texts = self.data["text"].tolist()
        tf_idf = TfidfVectorizer()
        tf_idf.fit(tokenized_texts)
        return dict(zip(tf_idf.get_feature_names(), tf_idf.idf_))


dev = TwitterDatasetTfIdf(dev_data, "text", "emotion", word2vec)

indexes = np.arange(len(dev))
np.random.shuffle(indexes)
example_indexes = indexes[::1000]

examples = {"features": [np.sum(dev[i]["feature"], axis=0) for i in example_indexes], 
            "targets": [dev[i]["target"] for i in example_indexes]}
print(len(examples["features"]))

pca = PCA(n_components=2)
examples["transformed_features"] = pca.fit_transform(examples["features"])

draw_vectors(
    examples["transformed_features"][:, 0], 
    examples["transformed_features"][:, 1], 
    color=[["red", "blue"][t] for t in examples["targets"]]
    )

train_size = math.ceil(len(dev) * 0.99)

train, valid = random_split(dev, [train_size, len(dev) - train_size])

train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, collate_fn=average_emb)
valid_loader = DataLoader(valid, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=average_emb)

vector_size = dev.word2vec.vector_size
num_classes = 1
lr = 1e-2
num_epochs = 2

model = LinearModel(vector_size=vector_size, num_classes=num_classes)
model = model.cuda()

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters())
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps)

best_metric = np.inf
for e in range(num_epochs):
    training(model, optimizer, criterion, train_loader, e, device, scheduler)
    log = testing(model, criterion, valid_loader, device)
    print(log)
    if log["Test Loss"] < best_metric:
        torch.save(model.state_dict(), "model.pt")
        best_metric = log["Test Loss"]

test = TwitterDatasetTfIdf(test_data, "text", "emotion", word2vec, weights=dev.weights)

test_loader = DataLoader(
    test, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=False,
    drop_last=False, 
    collate_fn=average_emb)

model.load_state_dict(torch.load("model.pt", map_location=device))

print(testing(model, criterion, test_loader, device=device))

vector_size = dev.word2vec.vector_size
num_classes = 2
lr = 5e-5
num_epochs = 5

model = ClassifierLSTM(vector_size, n_hidden=512, n_output=1, n_layers=2)
model = model.cuda()
criterion = nn.BCELoss()
total_steps = len(train_loader) * num_epochs
optimizer = AdamW(model.parameters(),lr=5e-5,eps=1e-8)
        
scheduler = CosineAnnealingLR(optimizer, T_max=100)
best_metric = np.inf
for e in range(num_epochs):
    training(model, optimizer, criterion, train_loader, e, device, scheduler=scheduler)
    log = testing(model, criterion, valid_loader, device)
    print(log)
    if log["Test Loss"] < best_metric:
        torch.save(model.state_dict(), "model.pt")
        best_metric = log["Test Loss"]

test = TwitterDatasetTfIdf(test_data, "text", "emotion", word2vec, weights=dev.weights)

test_loader = DataLoader(
    test, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=False,
    drop_last=False, 
    collate_fn=average_emb)

model.load_state_dict(torch.load("model.pt", map_location=device))

print(testing(model, criterion, test_loader, device=device))


