#!/usr/bin/env python
# coding: utf-8

# <img src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" width=500, height=450>
# <h3 style="text-align: center;"><b>Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ</b></h3>

# ---

# ### Задача определения частей речи, Part-Of-Speech Tagger (POS)

# Мы будем решать задачу определения частей речи (POS-теггинга) с помощью скрытой марковской модели (HMM).

# In[ ]:


import nltk
import pandas as pd
import numpy as np
from collections import OrderedDict, deque
from nltk.corpus import brown
import matplotlib.pyplot as plt


# Вам в помощь http://www.nltk.org/book/

# Загрузим brown корпус

# In[ ]:


nltk.download('brown')


# Существует множество наборов грамматических тегов, или тегсетов, например:
# * НКРЯ
# * Mystem
# * UPenn
# * OpenCorpora (его использует pymorphy2)
# * Universal Dependencies

# <b>Существует не одна система тегирования, поэтому будьте внимательны, когда прогнозируете тег слов в тексте и вычисляете качество прогноза. Можете получить несправедливо низкое качество вашего решения.

# На данный момент стандартом является **Universal Dependencies**. Подробнее про проект можно почитать [вот тут](http://universaldependencies.org/), а про теги — [вот тут](http://universaldependencies.org/u/pos/)

# In[ ]:


nltk.download('universal_tagset')


# <img src="https://4.bp.blogspot.com/-IcFli2wljs0/WrVCw3umY_I/AAAAAAAACYM/UJ_neoUAs3wF95dj2Ouf3BzxXzB_b2TbQCLcBGAs/s1600/postags.png">
# 

# Мы имеем массив предложений пар (слово-тег)

# In[ ]:


brown_tagged_sents = brown.tagged_sents(tagset="universal")
brown_tagged_sents


# Первое предложение

# In[ ]:


brown_tagged_sents[0]


# Все пары (слово-тег)

# In[ ]:


brown_tagged_words = brown.tagged_words(tagset='universal')
brown_tagged_words


# Проанализируйте данные, с которыми Вы работаете. Используйте `nltk.FreqDist()` для подсчета частоты встречаемости тега и слова в нашем корпусе. Под частой элемента подразумевается кол-во этого элемента в корпусе.

# In[ ]:


# Приведем слова к нижнему регистру
brown_tagged_words = list(map(lambda x: (x[0].lower(), x[1]), brown_tagged_words))


# In[ ]:


print('Кол-во предложений: ', len(brown_tagged_sents))
tags = [tag for (word, tag) in brown_tagged_words] # наши теги
words = [word for (word, tag) in brown_tagged_words] # наши слова

tag_num = pd.Series('''your code''').sort_values(ascending=False) # тег - кол-во тега в корпусе
word_num = pd.Series('''your code''').sort_values(ascending=False) # слово - кол-во слова в корпусе


# In[ ]:


tag_num


# In[ ]:


plt.figure(figsize=(12, 5))
plt.bar(tag_num.index, tag_num.values)
plt.title("Tag_frequency")
plt.show()


# In[ ]:


word_num[:5]


# In[ ]:


plt.figure(figsize=(12, 5))
plt.bar(word_num.index[:10], word_num.values[:10])
plt.title("Word_frequency")
plt.show()


# ### Вопрос 1:
# * Кол-во слова `cat` в корпусе?

# In[ ]:


'''your code'''


# ### Вопрос 2:
# * Самое популярное слово с самым популярным тегом? <br>(*сначала выбираете слова с самым популярным тегом, а затем выбираете самое популярное слово из уже выбранных*)

# In[ ]:


'''your code'''


# Впоследствии обучение моделей может занимать слишком много времени, работайте с подвыборкой, например, только текстами определенных категорий.

# Категории нашего корпуса:

# In[ ]:


brown.categories()


# Будем работать с категорией humor

# Cделайте случайное разбиение выборки на обучение и контроль в отношении 9:1. 

# In[ ]:


brown_tagged_sents = brown.tagged_sents(tagset="universal", categories='humor')
# Приведем слова к нижнему регистру
my_brown_tagged_sents = []
for sent in brown_tagged_sents:
    my_brown_tagged_sents.append(list(map(lambda x: (x[0].lower(), x[1]), sent)))
my_brown_tagged_sents = np.array(my_brown_tagged_sents)

from sklearn.model_selection import train_test_split
train_sents, test_sents = train_test_split('''your code''', random_state=0,)


# In[ ]:


len(train_sents)


# In[ ]:


len(test_sents)


# ### Метод максимального правдоподобия для обучения модели
# 
# * $\normalsize S = s_0, s_1, ..., s_N$ - скрытые состояния, то есть различные теги
# * $\normalsize O = o_0, o_1, ..., o_M$ - различные слова
# * $\normalsize a_{i,j} = p(s_j|s_i)$ - вероятность того, что, находясь в скрытом состоянии $s_i$, мы попадем в состояние $s_j$ (элемент матрицы $A$)
# * $\normalsize b_{k,j}=p(o_k|s_j)$ - вероятность того, что при скрытом состоянии $s_j$ находится слово $o_k$(элемент матрицы $B$)
# 
# $$\normalsize x_t \in O, y_t \in S$$
# $\normalsize (x_t, y_t)$ - слово и тег, стоящие на месте $t$ $\Rightarrow$ 
# * $\normalsize X$ - последовательность слов
# * $\normalsize Y$ - последовательность тегов
# 
# Требуется построить скрытую марковскую модель (class HiddenMarkovModel) и написать метод fit для настройки всех её параметров с помощью оценок максимального правдоподобия по размеченным данным (последовательности пар слово+тег):
# 
# - Вероятности переходов между скрытыми состояниями $p(y_t | y_{t - 1})$ посчитайте на основе частот биграмм POS-тегов.
# 
# 
# - Вероятности эмиссий наблюдаемых состояний $p(x_t | y_t)$ посчитайте на основе частот "POS-тег - слово".
# 
# 
# - Распределение вероятностей начальных состояний $p(y_0)$ задайте равномерным.
# 
# Пример $X = [x_0, x_1], Y = [y_0, y_1]$:<br><br>
# $$p(X, Y) = p(x_0, x_1, y_0, y_1) = p(y_0) \cdot p(x_0, x_1, y_1 | y_0) = p(y_0) \cdot p(x_0 | y_0) \cdot
# p(x_1, y_1 | x_0, y_0) = \\ = p(y_0) \cdot p(x_0 | y_0) \cdot p(y_1 | x_0, y_0) \cdot p(x_1 | x_0, y_0, y_1)
# = (\text{в силу условий нашей модели}) = \\ = p(y_0) \cdot p(x_0 | y_0) \cdot p(y_1 | y_0) \cdot p(x_1 | y_1) \Rightarrow$$ <br>
# Для последовательности длины $n + 1$:<br>
# $$p(X, Y) = p(x_0 ... x_{n - 1}, y_0 ... y_{n - 1}) \cdot p(y_n | y_{n - 1}) \cdot p(x_n | y_n)$$

# #### Алгоритм Витерби для применения модели
# 
# 
# Требуется написать метод .predict для определения частей речи на тестовой выборке. Чтобы использовать обученную модель на новых данных, необходимо реализовать алгоритм Витерби. Это алгоритм динамиеского программирования, с помощью которого мы будем находить наиболее вероятную последовательность скрытых состояний модели для фиксированной последовательности слов:
# 
# $$ \hat{Y} = \arg \max_{Y} p(Y|X) = \arg \max_{Y} p(Y, X) $$
# 
# Пусть $\normalsize Q_{t,s}$ - самая вероятная последовательность скрытых состояний длины $t$ с окончанием в состоянии $s$. $\normalsize q_{t, s}$ - вероятность этой последовательности.
# $$(1)\: \normalsize q_{t,s} = \max_{s'} q_{t - 1, s'} \cdot p(s | s') \cdot p(o_t | s)$$
# $\normalsize Q_{t,s}$ можно восстановить по argmax-ам.

# In[ ]:


class HiddenMarkovModel:    
    def __init__(self):
    
        pass
        
    def fit(self, train_tokens_tags_list):
        """
        train_tokens_tags_list: массив предложений пар слово-тег (выборка для train) 
        """
        tags = [tag for sent in train_tokens_tags_list
                for (word, tag) in sent]
        words = [word for sent in train_tokens_tags_list
                 for (word, tag) in sent]
        
        tag_num = pd.Series('''your code''').sort_index()
        word_num = pd.Series('''your code''').sort_values(ascending=False)
         
        self.tags = tag_num.index
        self.words = word_num.index
        
        A = pd.DataFrame({'{}'.format(tag) : [0] * len(tag_num) for tag in tag_num.index}, index=tag_num.index)
        B = pd.DataFrame({'{}'.format(tag) : [0] * len(word_num) for tag in tag_num.index}, index=word_num.index)
        
        # Вычисляем матрицу A и B по частотам слов и тегов
        
        # sent - предложение
        # sent[i][0] - i слово в этом предложении, sent[i][1] - i тег в этом предложении
        for sent in train_tokens_tags_list:
            for i in range(len(sent)):
                B.loc['''your code'''] += 1 # текущая i-пара слово-тег (обновите матрицу B аналогично A)
                if len(sent) - 1 != i: # для последнего тега нет следующего тега
                    A.loc[sent[i][1], sent[i + 1][1]] += 1 # пара тег-тег
                
        
        # переходим к вероятностям
        
        # нормируем по строке, то есть по всем всевозможным следующим тегам
        A = A.divide(A.sum(axis=1), axis=0)
        
        # нормируем по столбцу, то есть по всем всевозможным текущим словам
        B = B / np.sum(B, axis=0)
        
        self.A = A
        self.B = B
        
        return self
        
    
    def predict(self, test_tokens_list):
        """
        test_tokens_list : массив предложений пар слово-тег (выборка для test)
        """
        predict_tags = OrderedDict({i : np.array([]) for i in range(len(test_tokens_list))})
        
        for i_sent in range(len(test_tokens_list)):
            
            current_sent = test_tokens_list[i_sent] # текущее предложение
            len_sent = len(current_sent) # длина предложения 
            
            q = np.zeros(shape=(len_sent + 1, len(self.tags)))
            q[0] = 1 # нулевое состояние (равномерная инициализация по всем s)
            back_point = np.zeros(shape=(len_sent + 1, len(self.tags))) # # argmax
            
            for t in range(len_sent):
                
                # если мы не встречали такое слово в обучении, то вместо него будет 
                # самое популярное слово с самым популярным тегом (вопрос 2)
                if current_sent[t] not in self.words:
                    current_sent[t] = '''your code'''
                    
                # через max выбираем следующий тег
                for i_s in range(len(self.tags)):
                    
                    s = self.tags[i_s]
                    
                    # формула (1)
                    q[t + 1][i_s] = np.max(q['''your code'''] *
                        self.A.loc[:, '''your code'''] * 
                        self.B.loc[current_sent[t], s])
                    
                    # argmax формула(1)
                    
                    # argmax, чтобы восстановить последовательность тегов
                    back_point[t + 1][i_s] = (q['''your code'''] * self.A.loc[:, '''your code'''] * 
                        self.B.loc[current_sent[t],s]).reset_index()[s].idxmax() # индекс 
                    
            back_point = back_point.astype('int')
            
            # выписываем теги, меняя порядок на реальный
            back_tag = deque()
            current_tag = np.argmax(q[len_sent])
            for t in range(len_sent, 0, -1):
                back_tag.appendleft(self.tags[current_tag])
                current_tag = back_point[t, current_tag]
             
            predict_tags[i_sent] = np.array(back_tag)
        
        
        return predict_tags                 


# Обучите скрытую марковскую модель:

# In[ ]:


# my_model = ..,
'''your code'''


# Проверьте работу реализованного алгоритма на следующих модельных примерах, проинтерпретируйте результат.
# 
# - 'He can stay'
# - 'a cat and a dog'
# - 'I have a television'
# - 'My favourite character'

# In[ ]:


sents = [['He', 'can', 'stay'], ['a', 'cat', 'and', 'a', 'dog'], ['I', 'have', 'a', 'television'],
         ['My', 'favourite', 'character']]
'''your code'''


# ### Вопрос 3:
# * Какой тег вы получили для слова `can`?

# In[ ]:


'''your code'''


# ### Вопрос 4:
# * Какой тег вы получили для слова `favourite`?

# In[ ]:


'''your code'''


# Примените модель к отложенной выборке Брауновского корпуса и подсчитайте точность определения тегов (accuracy). Сделайте выводы. 

# In[ ]:


def accuracy_score(model, sents):
    true_pred = 0
    num_pred = 0

    for sent in sents:
        tags = '''your code'''
        words = '''your code'''

        '''your code'''

        true_pred += '''your code'''
        num_pred += '''your code'''
    print("Accuracy:", true_pred / num_pred * 100, '%')


# In[ ]:


accuracy_score(my_model, test_sents)


# ### Вопрос 5:
# * Какое качество вы получили(округлите до одного знака после запятой)?

# In[ ]:


'''your code'''


# ## DefaultTagger

# ### Вопрос 6:
# * Какое качество вы бы получили, если бы предсказывали любой тег, как самый популярный тег на выборке train(округлите до одного знака после запятой)?

# Вы можете испоьзовать DefaultTagger(метод tag для предсказания частей речи предложения)

# In[ ]:


from nltk.tag import DefaultTagger
default_tagger = DefaultTagger('''your code''')


# In[ ]:


'''your code'''


# ## NLTK, Rnnmorph

# Вспомним первый [семинар](https://colab.research.google.com/drive/1FHZVU6yJT61J8w1hALno0stD4VU36rit?usp=sharing) нашего курса. В том семинаре мы с вами работали c некоторыми библиотеками.
# 
# Не забудьте преобразовать систему тэгов из `'en-ptb' в 'universal'` с помощью функции `map_tag` или используйте `tagset='universal'`

# In[ ]:


from nltk.tag.mapping import map_tag


# In[ ]:


import nltk
nltk.download('averaged_perceptron_tagger')
# nltk.pos_tag(..., tagset='universal')


# In[ ]:


from rnnmorph.predictor import RNNMorphPredictor
predictor = RNNMorphPredictor(language="en")


# ### Вопрос 7:
# * Какое качество вы получили, используя каждую из двух библиотек? Сравните их результаты.
# 
# * Качество с библиотекой rnnmorph должно быть хуже, так как там используется немного другая система тэгов. Какие здесь отличия?

# In[ ]:


'''your code'''


# ## BiLSTMTagger

# ### Подготовка данных

# Изменим структуру данных

# In[ ]:


pos_data = [list(zip(*sent)) for sent in brown_tagged_sents]
print(pos_data[0])


# До этого мы писали много кода сами, теперь пора эксплуатировать pytorch

# In[ ]:


from torchtext.data import Field, BucketIterator
import torchtext

# наши поля
WORD = Field(lower=True)
TAG = Field(unk_token=None) # все токены нам извсетны

# создаем примеры
examples = []
for words, tags in pos_data:
    examples.append(torchtext.data.Example.fromlist([list(words), list(tags)], fields=[('words', WORD), ('tags', TAG)]))


# Вот один наш пример:

# In[ ]:


print(vars(examples[0]))


# Теперь формируем наш датасет

# In[ ]:


# кладем примеры в наш датасет
dataset = torchtext.data.Dataset('''your code''', fields=[('words', WORD), ('tags', TAG)])

train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.1, 0.1])

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")


# Построим словари. Параметр `min_freq` выберете сами. При построении словаря испольузем только **train**

# In[ ]:


WORD.build_vocab('''your code''', min_freq='''your code''')
TAG.build_vocab('''your code''')

print(f"Unique tokens in source (ru) vocabulary: {len(WORD.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TAG.vocab)}")

print(WORD.vocab.itos[::200])
print(TAG.vocab.itos)


# In[ ]:


print(vars(train_data.examples[9]))


# Посмотрим с насколько большими предложениями мы имеем дело

# In[ ]:


length = map(len, [vars(x)['words'] for x in train_data.examples])

plt.figure(figsize=[8, 4])
plt.title("Length distribution in Train data")
plt.hist(list(length), bins=20);


# Для обучения `BiLSTM` лучше использовать colab

# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# Для более быстрого и устойчивого обучения сгруппируем наши данные по батчам

# In[ ]:


# бьем нашу выборку на батч, не забывая сначала отсортировать выборку по длине
def _len_sort_key(x):
    return len(x.words)

BATCH_SIZE = 32

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device,
    sort_key=_len_sort_key
)


# In[ ]:


# посморим  на количество батчей
list(map(len, [train_iterator, valid_iterator, test_iterator]))


# ### Модель и её обучение

# Инициализируем нашу модель

# In[ ]:


class LSTMTagger(nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, dropout, bidirectional=False):
        super().__init__()
        
  
        self.embeddings = '''your code'''
        self.dropout = '''your code'''
        
        self.rnn = '''your code'''
        # если bidirectional, то предсказываем на основе конкатенации двух hidden
        self.tag = nn.Linear((1 + bidirectional) * hid_dim, output_dim)

    def forward(self, sent):
        
        #sent = [sent len, batch size] 
        
        # не забываем применить dropout к embedding
        embedded = '''your code'''

        output, _ = '''your code'''
        #output = [sent len, batch size, hid dim * n directions]

        prediction = '''your code'''
    
        return prediction
        
# параметры модели
INPUT_DIM = '''your code'''
OUTPUT_DIM = '''your code'''
EMB_DIM = '''your code'''
HID_DIM = '''your code'''
DROPOUT = '''your code'''
BIDIRECTIONAL = '''your code'''

model = LSTMTagger('''your code''').to(device)

# инициализируем веса
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)
        
model.apply(init_weights)


# Подсчитаем количество обучаемых параметров нашей модели

# In[ ]:


def count_parameters(model):
    return '''your code'''

print(f'The model has {count_parameters(model):,} trainable parameters')


# Погнали обучать

# In[ ]:


PAD_IDX = TAG.vocab.stoi['<pad>']
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

def train(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None):
    model.train()
    
    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        
       '''your code'''
        
        optimizer.zero_grad()
        
        output = model('''your code''')
        
        #tags = [sent len, batch size]
        #output = [sent len, batch size, output dim]
        
        output = '''your code'''
        tags = tags.view(-1)
        
        #tags = [sent len * batch size]
        #output = [sent len * batch size, output dim]
        
        loss = criterion('''your code''')
        
        loss.backward()
        
        # Gradient clipping(решение проблемы взрыва граденты), clip - максимальная норма вектора
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        
        optimizer.step()
        
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
            if valid_history is not None:
                ax[1].plot(valid_history, label='general valid history')
            plt.legend()
            
            plt.show()

        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    
    epoch_loss = 0
    
    history = []
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            '''your code'''

            output = model('''your code''')

            #tags = [sent len, batch size]
            #output = [sent len, batch size, output dim]

            output = '''your code'''
            tags = tags.view(-1)

            #tags = [sent len * batch size]
            #output = [sent len * batch size, output dim]

            loss = criterion('''your code''')
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[ ]:


import time
import math
import matplotlib
matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output

train_history = []
valid_history = []

N_EPOCHS = '''your code'''
CLIP = '''your code'''

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


# ### Применение модели

# In[ ]:


def accuracy_model(model, iterator):
    model.eval()
    
    true_pred = 0
    num_pred = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):

           '''your code'''

            output = model('''your code''')
            
            #output = [sent len, batch size, output dim]
            output = '''your code'''
            
            #output = [sent len, batch size]
            predict_tags = output.cpu().numpy()
            true_tags = tags.cpu().numpy()

            true_pred += np.sum((true_tags == predict_tags) & (true_tags != PAD_IDX))
            num_pred += np.prod(true_tags.shape) - (true_tags == PAD_IDX).sum()
        
    return round(true_pred / num_pred * 100, 3)


# In[ ]:


print("Accuracy:", accuracy_model(model, test_iterator), '%')


# Вы можете улучшить качество, изменяя параметры модели. Но чтобы добиться нужного качества, вам неообходимо взять все выборку, а не только категорию `humor`.

# In[ ]:


#brown_tagged_sents = brown.tagged_sents(tagset="universal")


# Вам неоходимо добиться качества не меньше, чем `accuracy = 93 %` 

# In[ ]:


best_model = LSTMTagger(INPUT_DIM, EMB_DIM, HID_DIM, OUTPUT_DIM, DROPOUT, BIDIRECTIONAL).to(device)
best_model.load_state_dict(torch.load('best-val-model.pt'))
assert accuracy_model(best_model, test_iterator) >= 93


# Пример решение нашей задачи:

# In[ ]:


def print_tags(model, data):
    model.eval()
    
    with torch.no_grad():
        words, _ = data
        example = torch.LongTensor([WORD.vocab.stoi[elem] for elem in words]).unsqueeze(1).to(device)
        
        output = model(example).argmax(dim=-1).cpu().numpy()
        tags = [TAG.vocab.itos[int(elem)] for elem in output]

        for token, tag in zip(words, tags):
            print(f'{token:15s}{tag}')


# In[ ]:


print_tags(model, pos_data[-1])


# ## Сравните результаты моделей HiddenMarkov, LstmTagger:
# * при обучение на маленькой части корпуса, например, на категории humor
# * при обучении на всем корпусе

# In[ ]:




