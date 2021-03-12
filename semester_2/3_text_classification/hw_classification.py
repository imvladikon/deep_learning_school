import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import numpy as np

try:
  from torchtext import datasets
  from torchtext.data import Field, LabelField
  from torchtext.data import BucketIterator
except:
  from torchtext.legacy import datasets
  from torchtext.legacy.data import Field, LabelField
  from torchtext.legacy.data import BucketIterator

from torchtext.vocab import Vectors, GloVe
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm.autonotebook import tqdm

import warnings
warnings.filterwarnings('ignore')

TEXT = Field(sequential=True, lower=True, include_lengths=True)  # Поле текста
LABEL = LabelField(dtype=torch.float)  # Поле метки

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train, test = datasets.IMDB.splits(TEXT, LABEL)  # загрузим датасет
train, valid = train.split(random_state=random.seed(SEED))  # разобьем на части

TEXT.build_vocab(train)
LABEL.build_vocab(train)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train, valid, test), 
    batch_size = 64,
    sort_within_batch = True,
    device = device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

class F1_Loss(nn.Module):
    '''
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_true, y_pred, return_pr=False):
        assert y_pred.ndim == 2, y_pred.shape
        assert y_true.ndim == 1, y_true.shape
        y_true = F.one_hot(y_true.long(), 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        if not return_pr:
          return 1 - f1.mean()
        else:
          return 1 - f1.mean(), precision.mean(), recall.mean()


@torch.no_grad()
def f1_score(y_true, y_pred):
  f1_loss = F1_Loss().to(device)
  return f1_loss(y_true, y_pred)

@torch.no_grad()
def test_evaluating(model, iterator, criterion):
    test_loss = 0.
    test_acc = 0.
    test_f1 = 0.
    test_precision = 0.
    test_recall = 0.
    f1_loss = F1_Loss().to(device)
    model.eval()
    for batch in iterator:
        if isinstance(batch.text, tuple):
          text, text_lengths = batch.text
          predictions = model(text, text_lengths)
        else:
          text = batch.text
          predictions = model(text)
        y_true = batch.label
        loss = criterion(predictions.squeeze(1), y_true)
        f1, precision, recall = f1_loss(y_true, predictions, return_pr=True)
        acc = binary_accuracy(predictions.squeeze(1), y_true)
        test_loss += loss.item()
        test_acc += acc.item()
        test_f1 += f1.item()
        test_precision += precision.item()
        test_recall += recall.item()
    return {
        "test_loss": test_loss / len(iterator), 
        "test_acc": test_acc / len(iterator), 
        "test_f1": test_f1 / len(iterator),
        "test_precision": test_precision / len(iterator),
        "test_recall": test_recall / len(iterator)
    }

def plot_learning_curves(metric_history, title=""):
    with plt.style.context('seaborn-whitegrid'):
      fig,ax = plt.subplots(1,2, figsize=(16, 6))
      train_history = pd.DataFrame(metric_history["train"]).reset_index()
      val_history = pd.DataFrame(metric_history["val"]).reset_index()
      train_history.plot(x="epoch", y="acc", ax=ax[0], color="r", label="acc_train") 
      val_history.plot(x="epoch", y="acc", ax=ax[0], color="b", label="acc_val")
      train_history.plot(x="epoch", y="loss", color="r", ax=ax[1], label="loss_train")
      val_history.plot(x="epoch", y="loss", color="b", ax=ax[1], label="loss_val")
      ax[0].set_title(f'Train Acc: {train_history.iloc[-1]["acc"]:.4f} Val Acc: {val_history.iloc[-1]["acc"]:.4f}')
      ax[1].set_title(f'Train Loss: {train_history.iloc[-1]["loss"]:.4f} Val Loss: {val_history.iloc[-1]["loss"]:.4f}')
      if not title:
        fig.suptitle(title)
      plt.show();

class RNNBaseline(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                        dropout=dropout, bidirectional=bidirectional)
        
        
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
        
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
           
        # workaround because of https://github.com/pytorch/pytorch/issues/43227

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded.cpu(), text_lengths.cpu()).to(device)
        
        # cell arg for LSTM, remove for GRU
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)  

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions] or [batch_size, hid dim * num directions]
            
        return self.fc(hidden)

vocab_size = len(TEXT.vocab)
emb_dim = 100
hidden_dim = 256
output_dim = 1
n_layers = 2
bidirectional = True
dropout = 0.2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
patience=3

model = RNNBaseline(
    vocab_size=vocab_size,
    embedding_dim=emb_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_layers=n_layers,
    bidirectional=bidirectional,
    dropout=dropout,
    pad_idx=PAD_IDX
).to(device)

optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.BCEWithLogitsLoss()
max_epochs = 20
min_loss = np.inf
max_f1 = 0
criterion=loss_func
cur_patience = 0
max_grad_norm = 2
metric_history = {"train":[], "val":[]} 


for epoch in range(1, max_epochs + 1):
    train_loss = 0.0
    train_acc = 0.0
    train_f1 = 0.0
    model.train()
    pbar = tqdm(enumerate(train_iter), total=len(train_iter), leave=False)
    pbar.set_description(f"Epoch {epoch}")
    for it, batch in pbar: 
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths)
        f1 = f1_score(batch.label, predictions)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        if max_grad_norm is not None:
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        train_loss += loss.item()
        train_acc += acc.item()
        train_f1 += f1.item()

    train_loss /= len(train_iter)
    train_acc /= len(train_iter)
    train_f1 /= len(train_iter)
    metric_history["train"].append({"epoch": epoch,"loss":train_loss, "acc":train_acc, "f1":train_f1})
    val_loss = 0.0
    val_acc = 0.0
    val_f1 = 0.0
    model.eval()
    pbar = tqdm(enumerate(valid_iter), total=len(valid_iter), leave=False)
    pbar.set_description(f"Epoch {epoch}")
    for it, batch in pbar:
        text, text_lengths = batch.text
        predictions = model(text, text_lengths)
        f1 = f1_score(batch.label, predictions)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        val_loss += loss.item()
        val_acc += acc.item()
        val_f1 += f1.item()
    val_loss /= len(valid_iter)
    val_acc /= len(valid_iter)
    val_f1 /= len(valid_iter)
    metric_history["val"].append({"epoch": epoch,"loss":val_loss, "acc":val_acc, "f1":val_f1})
    # if val_loss < min_loss:
    #     min_loss = val_loss
    #     best_model = model.state_dict()
    if val_f1 > max_f1:
        max_f1 = val_f1
        best_model = model.state_dict()
    else:
        cur_patience += 1
        if cur_patience == patience:
            cur_patience = 0
            break
    
    print('Epoch: {}, train loss: {}, val loss: {}, train acc: {}, val acc: {}, train f1: {}, val f1: {}'.format(epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1))
model.load_state_dict(best_model)

plot_learning_curves(metric_history)

metrics = test_evaluating(model, test_iter, criterion)
metrics["test_f1"]

TEXT = Field(sequential=True, lower=True, batch_first=True)  # batch_first тк мы используем conv  
LABEL = LabelField(batch_first=True, dtype=torch.float)

train, tst = datasets.IMDB.splits(TEXT, LABEL)
trn, vld = train.split(random_state=random.seed(SEED))

TEXT.build_vocab(trn)
LABEL.build_vocab(trn)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_iter, val_iter, test_iter = BucketIterator.splits(
        (trn, vld, tst),
        batch_sizes=(128, 256, 256),
        sort=False,
        sort_key= lambda x: len(x.src),
        sort_within_batch=False,
        device=device,
        repeat=False,
)

class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        out_channels,
        kernel_sizes,
        dropout=0.5,
        dim = 300
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv_0 = nn.Conv2d(in_channels = 1, out_channels = out_channels, 
                                kernel_size = (kernel_sizes[0], dim))
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = out_channels, 
                                kernel_size = (kernel_sizes[1], dim))
        self.conv_2 = nn.Conv2d(in_channels = 1, out_channels = out_channels, 
                                kernel_size = (kernel_sizes[2], dim))
        
        self.fc = nn.Linear(len(kernel_sizes) * out_channels, 1)
        
        self.dropout = nn.Dropout(dropout)
        # self.init_weights()
        
        
    def forward(self, text):
        
        embedded = self.embedding(text)
        
        embedded = embedded.unsqueeze(1)
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
            
        return self.fc(cat)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

kernel_sizes = [3, 4, 5]
vocab_size = len(TEXT.vocab)
out_channels=64
dropout = 0.5
dim = 300

model = CNN(vocab_size=vocab_size, emb_dim=dim, out_channels=out_channels,
            kernel_sizes=kernel_sizes, dropout=dropout).to(device)

min_loss = np.inf
max_f1 = 0
metric_history = {"train":[], "val":[]} 
cur_patience = 0

max_epochs = 30
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.BCEWithLogitsLoss()

for epoch in range(1, max_epochs + 1):
    train_loss = 0.0
    train_acc = 0.0
    train_f1 = 0.0
    model.train()
    pbar = tqdm(enumerate(train_iter), total=len(train_iter), leave=False)
    pbar.set_description(f"Epoch {epoch}")
    for it, batch in pbar: 
        optimizer.zero_grad()
        text = batch.text
        predictions = model(text)
        f1 = f1_score(batch.label, predictions)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += acc.item()
        train_f1 += f1.item()

    train_loss /= len(train_iter)
    train_acc /= len(train_iter)
    train_f1 /= len(train_iter)
    metric_history["train"].append({"epoch": epoch,"loss":train_loss, "acc":train_acc, "f1":train_f1})
    
    val_loss = 0.0
    val_acc = 0.0
    val_f1 = 0.0
    model.eval()
    pbar = tqdm(enumerate(val_iter), total=len(val_iter), leave=False)
    pbar.set_description(f"Epoch {epoch}")
    for it, batch in pbar:
        text = batch.text
        predictions = model(text)
        f1 = f1_score(batch.label, predictions)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        val_loss += loss.item()
        val_acc += acc.item()
        val_f1 += f1.item()
    
    val_loss /= len(val_iter)
    val_acc /= len(val_iter)
    val_f1 /= len(val_iter)
    metric_history["val"].append({"epoch": epoch,"loss":val_loss, "acc":val_acc, "f1":val_f1})
    if val_f1 > max_f1:
        max_f1 = val_f1
        best_model = model.state_dict()
    else:
        cur_patience += 1
        if cur_patience == patience:
            cur_patience = 0
            break
    
    print('Epoch: {}, train loss: {}, val loss: {}, train acc: {}, val acc: {}, train f1: {}, val f1: {}'.format(epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1))
model.load_state_dict(best_model)

plot_learning_curves(metric_history)

metrics = test_evaluating(model, test_iter, criterion)
metrics["test_f1"]

min_loss = np.inf
max_f1 = 0
metric_history = {"train":[], "val":[]} 
cur_patience = 0

max_epochs = 30
optimizer = torch.optim.Adam(model.parameters())
loss_func = F1_Loss()

for epoch in range(1, max_epochs + 1):
    train_loss = 0.0
    train_acc = 0.0
    train_f1 = 0.0
    model.train()
    pbar = tqdm(enumerate(train_iter), total=len(train_iter), leave=False)
    pbar.set_description(f"Epoch {epoch}")
    for it, batch in pbar: 
        optimizer.zero_grad()
        text = batch.text
        predictions = model(text)
        f1 = f1_score(batch.label, predictions)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += acc.item()
        train_f1 += f1.item()

    train_loss /= len(train_iter)
    train_acc /= len(train_iter)
    train_f1 /= len(train_iter)
    metric_history["train"].append({"epoch": epoch,"loss":train_loss, "acc":train_acc, "f1":train_f1})
    
    val_loss = 0.0
    val_acc = 0.0
    val_f1 = 0.0
    model.eval()
    pbar = tqdm(enumerate(val_iter), total=len(val_iter), leave=False)
    pbar.set_description(f"Epoch {epoch}")
    for it, batch in pbar:
        text = batch.text
        predictions = model(text)
        f1 = f1_score(batch.label, predictions)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        val_loss += loss.item()
        val_acc += acc.item()
        val_f1 += f1.item()
    
    val_loss /= len(val_iter)
    val_acc /= len(val_iter)
    val_f1 /= len(val_iter)
    metric_history["val"].append({"epoch": epoch,"loss":val_loss, "acc":val_acc, "f1":val_f1})
    if val_f1 > max_f1:
        max_f1 = val_f1
        best_model = model.state_dict()
    else:
        cur_patience += 1
        if cur_patience == patience:
            cur_patience = 0
            break
    
    print('Epoch: {}, train loss: {}, val loss: {}, train acc: {}, val acc: {}, train f1: {}, val f1: {}'.format(epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1))
model.load_state_dict(best_model)


metrics = test_evaluating(model, test_iter, criterion)
metrics["test_f1"]

!pip install -q captum

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

PAD_IND = TEXT.vocab.stoi['pad']

token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
lig = LayerIntegratedGradients(model, model.embedding)

def forward_with_softmax(inp):
    logits = model(inp)
    return torch.softmax(logits, 0)[0][1]

def forward_with_sigmoid(input):
    return torch.sigmoid(model(input))


# accumalate couple samples in this array for visualization purposes
vis_data_records_ig = []

def interpret_sentence(model, sentence, min_len = 7, label = 0):
    model.eval()
    text = [tok for tok in TEXT.tokenize(sentence)]
    if len(text) < min_len:
        text += ['pad'] * (min_len - len(text))
    indexed = [TEXT.vocab.stoi[t] for t in text]

    model.zero_grad()

    input_indices = torch.tensor(indexed, device=device)
    input_indices = input_indices.unsqueeze(0)
    
    # input_indices dim: [sequence_length]
    seq_length = min_len

    # predict
    pred = forward_with_sigmoid(input_indices).item()
    pred_ind = round(pred)

    # generate reference indices for each sample
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig, delta = lig.attribute(input_indices, reference_indices, \
                                           n_steps=5000, return_convergence_delta=True)

    print('pred: ', LABEL.vocab.itos[pred_ind], '(', '%.2f'%pred, ')', ', delta: ', abs(delta))

    add_attributions_to_visualizer(attributions_ig, text, pred, pred_ind, label, delta, vis_data_records_ig)
    
def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            LABEL.vocab.itos[pred_ind],
                            LABEL.vocab.itos[label],
                            LABEL.vocab.itos[1],
                            attributions.sum(),       
                            text,
                            delta))

interpret_sentence(model, 'It was a fantastic performance !', label=1)
interpret_sentence(model, 'Best film ever', label=1)
interpret_sentence(model, 'Such a great show!', label=1)
interpret_sentence(model, 'It was a horrible movie', label=0)
interpret_sentence(model, 'I\'ve never watched something as bad', label=0)
interpret_sentence(model, 'It is a disgusting movie!', label=0)

interpret_sentence(model, 'It was not a bad movie!', label=1)
interpret_sentence(model, 'I would like watch this movie again', label=1)

print('Visualize attributions based on Integrated Gradients')
visualization.visualize_text(vis_data_records_ig)

try:
  # tqdm newline issue: https://stackoverflow.com/questions/41707229/tqdm-printing-to-newline
  tqdm._instances.clear()
except:
  pass

TEXT.build_vocab(trn, vectors="glove.6B.300d")
# подсказка: один из импортов пока не использовался, быть может он нужен в строке выше :)
LABEL.build_vocab(trn)
word_embeddings = TEXT.vocab.vectors

kernel_sizes = [3, 4, 5]
vocab_size = len(TEXT.vocab)
dropout = 0.5
dim = 300

train, tst = datasets.IMDB.splits(TEXT, LABEL)
trn, vld = train.split(random_state=random.seed(SEED))

device = "cuda" if torch.cuda.is_available() else "cpu"

train_iter, val_iter, test_iter = BucketIterator.splits(
        (trn, vld, tst),
        batch_sizes=(128, 256, 256),
        sort=False,
        sort_key= lambda x: len(x.src),
        sort_within_batch=False,
        device=device,
        repeat=False,
)

model = CNN(vocab_size=vocab_size, emb_dim=dim, out_channels=64,
            kernel_sizes=kernel_sizes, dropout=dropout).to(device)

word_embeddings = TEXT.vocab.vectors

prev_shape = model.embedding.weight.shape

model.embedding.weight.data.copy_(word_embeddings)

assert prev_shape == model.embedding.weight.shape

optimizer = torch.optim.Adam(model.parameters())

min_loss = np.inf
max_f1 = 0
metric_history = {"train":[], "val":[]} 
cur_patience = 0

max_epochs = 30
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.BCEWithLogitsLoss()

for epoch in range(1, max_epochs + 1):
    train_loss = 0.0
    train_acc = 0.0
    train_f1 = 0.0
    model.train()
    pbar = tqdm(enumerate(train_iter), total=len(train_iter), leave=False)
    pbar.set_description(f"Epoch {epoch}")
    for it, batch in pbar: 
        optimizer.zero_grad()
        text = batch.text
        predictions = model(text)
        f1 = f1_score(batch.label, predictions)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += acc.item()
        train_f1 += f1.item()

    train_loss /= len(train_iter)
    train_acc /= len(train_iter)
    train_f1 /= len(train_iter)
    metric_history["train"].append({"epoch": epoch,"loss":train_loss, "acc":train_acc, "f1":train_f1})
    
    val_loss = 0.0
    val_acc = 0.0
    val_f1 = 0.0
    model.eval()
    pbar = tqdm(enumerate(val_iter), total=len(val_iter), leave=False)
    pbar.set_description(f"Epoch {epoch}")
    for it, batch in pbar:
        text = batch.text
        predictions = model(text)
        f1 = f1_score(batch.label, predictions)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        val_loss += loss.item()
        val_acc += acc.item()
        val_f1 += f1.item()
    
    val_loss /= len(val_iter)
    val_acc /= len(val_iter)
    val_f1 /= len(val_iter)
    metric_history["val"].append({"epoch": epoch,"loss":val_loss, "acc":val_acc, "f1":val_f1})
    if val_f1 > max_f1:
        max_f1 = val_f1
        best_model = model.state_dict()
    else:
        cur_patience += 1
        if cur_patience == patience:
            cur_patience = 0
            break
    
    print('Epoch: {}, train loss: {}, val loss: {}, train acc: {}, val acc: {}, train f1: {}, val f1: {}'.format(epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1))
model.load_state_dict(best_model)

plot_learning_curves(metric_history)

metrics = test_evaluating(model, test_iter, criterion)
metrics["test_f1"]

PAD_IND = TEXT.vocab.stoi['pad']

token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
lig = LayerIntegratedGradients(model, model.embedding)
vis_data_records_ig = []

interpret_sentence(model, 'It was a fantastic performance !', label=1)
interpret_sentence(model, 'Best film ever', label=1)
interpret_sentence(model, 'Such a great show!', label=1)
interpret_sentence(model, 'It was a horrible movie', label=0)
interpret_sentence(model, 'I\'ve never watched something as bad', label=0)
interpret_sentence(model, 'It is a disgusting movie!', label=0)

print('Visualize attributions based on Integrated Gradients')
visualization.visualize_text(vis_data_records_ig)
