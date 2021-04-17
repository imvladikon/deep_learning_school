%%capture
!pip install transformers
!pip install git+https://github.com/PyTorchLightning/metrics

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import Dataset, random_split
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          AutoTokenizer,
                          AutoModelForSequenceClassification,
                          BertConfig,
                          AdamW, 
                          get_linear_schedule_with_warmup)

import torchmetrics

df = pd.read_csv(
    'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv',
    delimiter='\t',
    header=None
)
print(df.shape)
df.head()

# For DistilBERT, Load pretrained model/tokenizer:

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# look at the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

from termcolor import colored

colors = ['red', 'green', 'blue', 'yellow']

def model_structure(layer, margin=0, item_color=0):
    for name, next_layer in layer.named_children():

        next = (0 if not list(next_layer.named_children()) else 1)
        print(colored(' ' * margin + name, colors[item_color]) + ':' * next)
        model_structure(next_layer, margin + len(name) + 2, (item_color + 1) % 4)

model_structure(model)

df.columns = ["review", "label"]

def get_max_len(values):
    max_len = 0
    for value in values:
        if len(value) > max_len:
            max_len = len(value)
    return max_len

MAX_LEN = get_max_len(tokenizer.batch_encode_plus(df["review"].values).input_ids)

class ReviewsDataset(Dataset):
    def __init__(self, reviews, tokenizer, labels):
        self.labels = labels
        # tokenized reviews
        ret: transformers.tokenization_utils_base.BatchEncoding = tokenizer.batch_encode_plus(reviews, padding=True, max_length=MAX_LEN, return_tensors="pt")
        self.input_ids = ret.input_ids
        self.attention_mask = ret.attention_mask
        
    def __getitem__(self, idx):
        # batch_encode_plus возвращает уже готовую маску, давайте ее и использовать
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx], "attention_mask": self.attention_mask[idx]}

    def __len__(self):
        return len(self.labels)

dataset = ReviewsDataset(df["review"].values, tokenizer, df["label"].values)

# DON'T CHANGE, PLEASE
train_size, val_size = int(.8 * len(dataset)), int(.1 * len(dataset))
torch.manual_seed(2) 
train_data, valid_data, test_data = random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

from torch.utils.data import Sampler

class ReviewsSampler(Sampler):
    def __init__(self, subset, batch_size=32):
        self.batch_size = batch_size
        self.subset = subset

        self.indices = subset.indices
        # tokenized for our data
        self.input_ids = np.array(subset.dataset.input_ids)[self.indices]

    def __iter__(self):

        batch_idx = []
        # index in sorted data
        for index in np.argsort(list(map(len, self.input_ids))):
            batch_idx.append(index)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []

        if len(batch_idx) > 0:
            yield batch_idx

    def __len__(self):
        return len(self.subset.dataset) 

from torch.utils.data import DataLoader

# batch_encode_plus сделал это за нас

# def get_padded(values):
#     max_len = 0
#     for value in values:
#         if len(value) > max_len:
#             max_len = len(value)

#     padded = np.array([value + [0]*(max_len-len(value)) for value in values])

#     return padded

# def collate_fn(batch):
#     print(batch)

#     inputs = []
#     labels = []
#     attention_mask = []
#     for elem in batch:
#         inputs.append(elem['tokenized'])
#         labels.append(elem['label'])
#         attention_mask.append(elem['attention_mask'])

#     inputs = get_padded(inputs)

#     return {"inputs": torch.tensor(inputs), "labels": torch.FloatTensor(labels), 'attention_mask' : torch.tensor(attention_mask)}

# train_loader = DataLoader(train_data, batch_sampler=ReviewsSampler(train_data), collate_fn=collate_fn)
# valid_loader = DataLoader(valid_data, batch_sampler=ReviewsSampler(valid_data), collate_fn=collate_fn)
# test_loader = DataLoader(test_data, batch_sampler=ReviewsSampler(test_data), collate_fn=collate_fn)

train_loader = DataLoader(train_data, batch_sampler=ReviewsSampler(train_data))
valid_loader = DataLoader(valid_data, batch_sampler=ReviewsSampler(valid_data))
test_loader = DataLoader(test_data, batch_sampler=ReviewsSampler(test_data))

# я изменил исходное tokenized на input_ids просто ради удобства
next(iter(train_loader))

from tqdm.notebook import tqdm

@torch.no_grad()
def get_xy(loader):
    features = []
    labels = []

    for batch in tqdm(loader, total=len(loader)):
        
        # don't forget about .to(device)
        output = model(**{k:v.to(device) for k,v in batch.items() if k!="labels"})
        if isinstance(output, tuple):
          last_hidden_states = output,
        else:
          last_hidden_states = output.last_hidden_state
        features.append(last_hidden_states.cpu())
        labels.append(batch["labels"])

    features = torch.cat([elem[:, 0, :] for elem in features], dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    return features, labels

train_features, train_labels = get_xy(train_loader)
valid_features, valid_labels = get_xy(valid_loader)
test_features, test_labels = get_xy(test_loader)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels)

from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import DistilBertPreTrainedModel
from torch.nn import CrossEntropyLoss

class BertClassifier(DistilBertPreTrainedModel):  
    def __init__(self, pretrained_model, dropout=0.1):
        super().__init__(pretrained_model.config)
        config = pretrained_model.config
        self.num_labels = config.num_labels

        self.bert = pretrained_model
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        # for param in self.bert.parameters():
        #   param.requires_grad = False
        # self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None
    ):
        distilbert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = self.relu(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup

# DON'T CHANGE
model = model_class.from_pretrained(pretrained_weights).to(device)
bert_clf = BertClassifier(model).to(device)
# you can change
# optimizer = optim.Adam(bert_clf.parameters(), lr=2e-5)
# criterion = nn.BCELoss()

optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                  )

N_EPOCHS = 3
total_steps = len(train_loader) * N_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

def train(epoch, model, iterator, optimizer, clip, criterion=None, train_history=None, valid_history=None, scheduler=None):
    model.train()
    
    # мы ленивые и лень посчитать среднее по классам, будем тянуть целый пакет для этого;)
    acc_metric = torchmetrics.Accuracy()

    epoch_loss = 0
    epoch_acc = 0
    history = []
    for i, batch in enumerate(iterator):
        
        # don't forget about .to(device)
        input_tensors = {k:v.to(device) for k, v in batch.items()}
        output = model(**input_tensors)
        loss = output.loss
        logits = output.logits
        
        optimizer.zero_grad()

        # loss = criterion(output, labels)  --> мы его уже посчитали внутри модели
        loss.backward()
        acc_batch = acc_metric(torch.argmax(logits, dim = -1).cpu(), input_tensors["labels"].cpu()) 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
          scheduler.step()
        
        epoch_loss += loss.item()
        history.append(epoch_loss)
        if (i+1)%10==0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            
            clear_output(True)
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None and len(train_history)>0:
                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None and len(train_history)>0:
                ax[1].plot(valid_history, label='general valid history')
            plt.legend()
            
            plt.show()

    epoch_acc = acc_metric.compute()
    return epoch_loss / (i + 1), epoch_acc

@torch.no_grad()
def evaluate(epoch, model, iterator, criterion=None):
    
    model.eval()
    
    epoch_loss = 0
    acc_metric = torchmetrics.Accuracy()
    history = []
    
    for i, batch in enumerate(iterator):

        input_tensors = {k:v.to(device) for k, v in batch.items()}
        output = model(**input_tensors)
        loss = output.loss
        logits = output.logits
   
        # loss = criterion(output, labels) -- уже посчитали
        acc_batch = acc_metric(torch.argmax(logits, dim = -1).cpu(), input_tensors["labels"].cpu()) 
        
        epoch_loss += loss.item()
        
    return epoch_loss / (i + 1), acc_metric.compute()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

import time
import math
import matplotlib
matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import clear_output

metrics_history = []
train_history = []
valid_history = []

CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss, train_acc = train(epoch, bert_clf, train_loader, optimizer, CLIP, train_history = train_history, valid_history=valid_history, scheduler=scheduler)
    valid_loss, valid_acc = evaluate(epoch, bert_clf, valid_loader)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(bert_clf.state_dict(), 'best-val-model.pt')
    
    metrics_history.append({"epoch":epoch, "loss":train_loss, "acc": train_acc, "phase": "train"})
    metrics_history.append({"epoch":epoch, "loss":valid_loss, "acc": valid_acc, "phase": "val"})
    
    train_history.append(train_loss)
    valid_history.append(valid_loss)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Train Acc.: {train_acc:7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} |  Val. Acc.: {valid_acc:7.3f}')
    # а не должно быть math.exp(-train_loss) если имелась ввиду аппроксимация accuracy?

best_model = BertClassifier(model).to(device)
best_model.load_state_dict(torch.load('best-val-model.pt'))

pred_labels = []
true_labels = []

best_model.eval()
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_loader)):
        
        input_tensors = {k:v.to(device) for k, v in batch.items()}
        output = best_model(**input_tensors)
        loss = output.loss
        logits = output.logits

        true_labels.append(input_tensors["labels"].detach().cpu().numpy())
        pred_labels.append(torch.argmax(logits, dim = -1).detach().cpu().numpy())


from sklearn.metrics import accuracy_score

true_labels = np.concatenate(true_labels, axis=0)
pred_labels = np.concatenate(pred_labels, axis=0)
accuracy_score(true_labels, pred_labels)

assert accuracy_score(true_labels, pred_labels) >= 0.86

# we have the same tokenizer
# new_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
new_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device)

pred_labels = []
true_labels = []

new_model.eval()
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_loader)):

        input_tensors = {k:v.to(device) for k, v in batch.items()}
        output = new_model(**input_tensors)
        loss = output.loss
        logits = output.logits

        true_labels.append(input_tensors["labels"].detach().cpu().numpy())
        pred_labels.append(torch.argmax(logits, dim = -1).detach().cpu().numpy())

true_labels = np.concatenate(true_labels, axis=0)
pred_labels = np.concatenate(pred_labels, axis=0)
accuracy_score(true_labels, pred_labels)

model_structure(new_model)
