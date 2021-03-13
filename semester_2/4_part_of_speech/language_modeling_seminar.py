!pip install -q datasets

!wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
!unzip -p python.zip python/final/jsonl/train/python_train_0.jsonl.gz > train.jsonl.gz
!unzip -p python.zip python/final/jsonl/test/python_test_0.jsonl.gz > test.jsonl.gz

# decompress this gzip file
!gzip -d train.jsonl.gz
!gzip -d test.jsonl.gz

from datasets import load_dataset  
dataset = load_dataset(
    "json",
    data_files=[
        "train.jsonl",
    ],
)

dataset

from collections import Counter


vocab_size = 40000
stats = Counter()

for item in dataset["train"]:
    stats.update(item["code_tokens"])
tokens = dict(stats.most_common(vocab_size)).keys()

stats.most_common(20)

PAD = 0
UNK = 1
EOS = 2

token2idx = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}

for idx, token in enumerate(tokens):
    token2idx[token] = idx + 3

def encode(token):
    if token in token2idx.keys():
        return token2idx[token]
    return UNK

dataset = dataset.map(
    lambda item: {
        "features": [encode(token) for token in item["code_tokens"]] + [EOS]
    }
)

import numpy as np
from collections import Counter, defaultdict

from tqdm.notebook import tqdm


class NGramModel(object):
    """
    Структура этой реализации n-граммной модели следующая:
    self.ngrams – словарь, который на каждый (token_0, ..., token_(n-1)) – n-1 tuple из токенов
        хранит частоту появления следующего токена. Для подсчета числа токенов воспользуемся
        Counter
    self.tokenize_func – функция токенизации текста. С её помощью будем получать токены.
    """
    def __init__(self, n=2):
        self.ngrams = defaultdict(Counter)
        self.n = n
        self.tokenize_func = None
        
    def compute_ngrams(self, dataset):
        self.ngrams = defaultdict(Counter)
        for row in tqdm(dataset):
            ngram = [PAD] * self.n
            for token in row["features"]:
                ngram[:-1] = ngram[1:]
                ngram[-1] = token
                self.ngrams[tuple(ngram[:-1])].update([ngram[-1]])
            
    def get_log_probs(self, prefix, min_log_pr=-15):
        """
        Функция, которая будет возвращать логарифмы частот появления токенов
        """
        if len(prefix) < self.n - 1:
            prefix = [PAD] * (self.n - len(prefix) - 1) + prefix
        else:
            prefix = prefix[-self.n + 1:]
        possible_ends = self.ngrams[tuple(prefix)]
        sum_freq = np.log(sum(possible_ends[e] for e in possible_ends))
        return {e: np.log(possible_ends[e]) - sum_freq for e in possible_ends}
    
    def sample(self, prefix):
        possible_ends = self.get_log_probs(prefix)
        if len(possible_ends) > 0:
            end = np.random.choice(list(possible_ends.keys()), p=np.exp(list(possible_ends.values())))
            return end
        return EOS

n_gram_model = NGramModel(n=5)

n_gram_model.compute_ngrams(dataset["train"])

idx2token = {idx: token for token, idx in token2idx.items()}

prefix = ["def", "train", "("]
encoded_prefix = [token2idx[token] for token in prefix]
length=100

for i in range(length):
    cur_token = n_gram_model.sample(encoded_prefix)
    if cur_token == EOS:
        break
    encoded_prefix += [cur_token]


decoded_text = [idx2token[idx] for idx in encoded_prefix]
print(" ".join(decoded_text))

test_dataset = load_dataset(
    "json",
    data_files=[
        "test.jsonl",
    ],
)

max_seq_len=128

test_dataset = test_dataset.map(
    lambda item: {
        "features": [encode(token) for token in item["code_tokens"]][:max_seq_len-1] + [EOS]
    }
)

def count_perplexity(model, dataset, max_iter_num: int = 1000):
    entropy = 0
    iter_num = 0
    num_words = 0
    for item in tqdm(dataset, total=min(max_iter_num, len(dataset))):
        output_so_far = [item["features"][0]]

        for token in item["features"][1:]:
            num_words += 1
            try:
                log_probs = model.get_log_probs(output_so_far)
                entropy += -log_probs[token]
            except KeyError:
                entropy += np.log(-10)
            output_so_far.append(token)
        iter_num += 1
        if iter_num > max_iter_num:
            break
    mean_entropy = entropy / num_words
    return np.e ** mean_entropy

count_perplexity(n_gram_model, test_dataset["train"])

dataset.set_format(type="torch", columns=["features"])
test_dataset.set_format(type="torch", columns=["features"])

def collate_fn(batch):
    batch = batch[0]
    max_len = max(len(f_t) for f_t in batch["features"])
    input_embeds = torch.zeros((len(batch["features"]), max_len), dtype=torch.long)
    for idx, row in enumerate(batch["features"]):
        input_embeds[idx][:len(row)] += row
    return {
        "features": input_embeds,
    }

from torch.utils.data import Sampler


class TextSampler(Sampler):
    def __init__(self, sampler, batch_size_tokens=1e4):
        self.sampler = sampler
        self.batch_size_tokens = batch_size_tokens

    def __iter__(self):
        batch = []
        max_len = 0
        for ix in self.sampler:
            row = self.sampler.data_source[ix]
            max_len = max(max_len, len(row["features"]))
            if (len(batch) + 1) * max_len > self.batch_size_tokens:
                yield batch
                batch = []
                max_len = len(row["features"])
            batch.append(ix)
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return len(self.sampler)

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, random_split


train_sampler = RandomSampler(dataset["train"])
valid_sampler = SequentialSampler(test_dataset["train"])

loaders = {
    "train": DataLoader(
        dataset["train"], 
        collate_fn=collate_fn, 
        sampler=TextSampler(train_sampler,)
    ),
    "valid": DataLoader(
        test_dataset["train"],
        collate_fn=collate_fn, 
        sampler=TextSampler(
            valid_sampler, 
        )
    )
}

import torch
import torch.nn as nn


class CNNLM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=3, kernel_size: int = 5):
        super().__init__()
        
        self.emb = nn.Embedding(vocab_size, emb_size)
        layers = []
        for layer_idx in range(num_layers):
            layers.append(nn.ZeroPad2d((kernel_size-1, 0, 0, 0)))
            if layer_idx == 0:
                layers.append(nn.Conv1d(emb_size, hidden_size, kernel_size=kernel_size))
            else:
                layers.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size))
        self.conv_layers = nn.Sequential(*layers)
        self.receptive_field = kernel_size + (kernel_size-1)*(num_layers-1)
        self.pred = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        embed = self.emb(input_ids)
        embed = embed.permute(0, 2, 1)
        features = self.conv_layers(embed)
        features = features.permute(0, 2, 1)
        logits = self.pred(features)
        return logits

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = CNNLM(len(tokens) + 3, 300, 100, num_layers=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

from tqdm.notebook import tqdm, trange


def train(
    num_epochs: int, 
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = None
):
    for epoch in trange(num_epochs):
        pbar = tqdm(train_loader, leave=False, total=len(train_loader)//20)
        pbar.set_description("Train epoch")
        model.train()
        for batch in pbar:
            optimizer.zero_grad()
            features = batch["features"].to(device)
            predictions = model(features[:, :-1])
            loss = criterion(
                predictions.reshape(-1, predictions.size(-1)),
                features[:, 1:].reshape(-1)
            )
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        model.eval()
        mean_loss = 0
        pbar = tqdm(valid_loader, leave=False, total=len(valid_loader)//100)
        pbar.set_description("Valid epoch")
        num_iter=0
        for batch in pbar:
            features = batch["features"].to(device)
            with torch.no_grad():
                predictions = model(features[:, :-1])
                loss = criterion(
                    predictions.reshape(-1, predictions.size(-1)),
                    features[:, 1:].reshape(-1)
                )
            mean_loss += loss.item()
            num_iter += 1
        mean_loss /= num_iter
        print(f"Epoch: {epoch}; mean loss: {mean_loss}; perplexity: {np.exp(mean_loss)}")
            

train(
    num_epochs=1,
    model=model, 
    train_loader=loaders["train"],
    valid_loader=loaders["valid"],
    criterion=criterion,
    optimizer=optimizer,
)

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ipywidgets import interactive
from IPython import display

sns.set(style="whitegrid", font_scale=1.4)

sample = np.random.randn(10)
def plot_temperature(T: float = 1.0):
    plt.figure(figsize=(12, 8))
    plt.title(f"Temperature = {T}")
    probs = np.exp(sample / T) / sum(np.exp(sample / T))
    plt.bar(range(10), probs)
    plt.xlabel("tokens")
    plt.ylabel("probs")
    plt.show()


v = interactive(
    plot_temperature, T=(0.02, 10)
)

display.display(v)

from typing import List
from torch.distributions import Categorical

@torch.no_grad()
def generate(
    prefix, model, length: int = 100, receptive_field: int = 5, T: float = 1.
) -> List[int]:
    prefix = torch.from_numpy(prefix)
    prefix = prefix.unsqueeze(0).to(device)
    model.eval()
    for iter_idx in range(length):
        preds = model(prefix[:, -receptive_field:])
        probs = torch.softmax(preds[:, -1]/T, dim=-1)
        distribution = Categorical(probs)
        sampled = distribution.sample()
        if sampled.item() == EOS:
            break
        prefix = torch.cat((prefix, sampled.unsqueeze(0)), dim=1)
    return prefix

prefix = ["def", "train", "("]
encoded_prefix = np.array([token2idx[t] for t in prefix])


for t in np.logspace(0.002, 1, 10):
    generated = generate(
        encoded_prefix, 
        model, 
        receptive_field=model.receptive_field, 
        length=20,
        T=t-1
    )
    print(f"Temperature: {t-1}")
    print(" ".join([idx2token[idx] for idx in generated.cpu().numpy().flatten()]))

class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.pred = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        embs = self.emb(input_ids)
        output, _ = self.lstm(embs)
        return self.pred(output)

model = LSTM(len(token2idx), 300, 50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

train(
    num_epochs=1,
    model=model,
    train_loader=loaders["train"],
    valid_loader=loaders["valid"],
    criterion=criterion,
    optimizer=optimizer,
)

prefix = ["def", "train", "("]
encoded_prefix = np.array([token2idx[t] for t in prefix])

generated = generate(encoded_prefix, model)

prefix = ["def", "train", "("]
encoded_prefix = np.array([token2idx[t] for t in prefix])


for t in np.logspace(0.002, 1, 10):
    generated = generate(
        encoded_prefix, 
        model, 
        receptive_field=20, 
        length=20,
        T=t-1
    )
    print(f"Temperature: {t-1}")
    print(" ".join([idx2token[idx] for idx in generated.cpu().numpy().flatten()]))


