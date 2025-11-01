#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## **Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ**

# # Путешествие по Спрингфилду.
# 
# 
# Сегодня вам предстоить помочь телекомпании FOX  в обработке их контента. Как вы знаете сериал Симсоны идет на телеэкранах более 25 лет и за это время скопилось очень много видео материала. Персоонажи менялись вместе с изменяющимися графическими технологиями   и Гомер 2018 не очень похож на Гомера 1989. Нашей задачей будет научиться классифицировать персонажей проживающих в Спрингфилде. Думаю, что нет смысла представлять каждого из них в отдельности.
# 
# 
# 
#  ![alt text](https://vignette.wikia.nocookie.net/simpsons/images/5/5a/Spider_fat_piglet.png/revision/latest/scale-to-width-down/640?cb=20111118140828)
# 
# 

# ### Установка зависимостей

# In[1]:


get_ipython().run_cell_magic('capture', '', '%%bash\npip install -U torch torchvision kaggle==1.5.9 imagehash\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%%bash\n\ngdown https://drive.google.com/uc?id=1wvBBnfl1OyL5Q99p3Qpz9zegZpq-UF_L\nunzip journey-springfield.zip -d .\n')


# In[5]:


import glob2 as glob
import os
import pandas as pd
import numpy as np


# In[12]:


r=[]
for f in glob.glob("/content/train/simpsons_dataset/*"):
  r.append({"person":os.path.basename(f), "count": len(glob.glob(f"{f}/*.jpg"))})
pd.DataFrame(r).sort_values(by="count")


# In[14]:


import torch
import torchvision
import numpy as np

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# In[15]:


import PIL
print(PIL.__version__)
print(torch.__version__)


# В нашем тесте будет 990 картнок, для которых вам будет необходимо предсказать класс.

# In[16]:


import pickle
import numpy as np
from skimage import io

from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path

from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from matplotlib import colors, pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# в sklearn не все гладко, чтобы в colab удобно выводить картинки 
# мы будем игнорировать warnings
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


# In[17]:


# разные режимы датасета 
DATA_MODES = ['train', 'val', 'test']
# все изображения будут масштабированы к размеру 224x224 px
RESCALE_SIZE = 224
# работаем на видеокарте
DEVICE = torch.device("cuda")


# Ниже мы исспользуем враппер над датасетом для удобной работы. Вам стоит понимать, что происходит с LabelEncoder и  с torch.Transformation. 
# 
# ToTensor конвертирует  PIL Image с параметрами в диапазоне [0, 255] (как все пиксели) в FloatTensor размера (C x H x W) [0,1] , затем производится масштабирование:
# $input = \frac{input - \mu}{\text{standard deviation}} $, <br>       константы - средние и дисперсии по каналам на основе ImageNet
# 
# 
# Стоит также отметить, что мы переопределяем метод __getitem__ для удобства работы с данной структурой данных.
#  Также используется LabelEncoder для преобразования строковых меток классов в id и обратно. В описании датасета указано, что картинки разного размера, так как брались напрямую с видео, поэтому следуем привести их к одному размер (это делает метод  _prepare_sample) 

# In[18]:


class SimpsonsDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    def __init__(self, files, mode):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        # режим работы
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)


# In[19]:


def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


# In[20]:


TRAIN_DIR = Path('train/simpsons_dataset')
TEST_DIR = Path('testset/testset')

train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
test_files = sorted(list(TEST_DIR.rglob('*.jpg')))


# In[21]:


from sklearn.model_selection import train_test_split

train_val_labels = [path.parent.name for path in train_val_files]
train_files, val_files = train_test_split(train_val_files, test_size=0.25, \
                                          stratify=train_val_labels)


# In[22]:


val_dataset = SimpsonsDataset(val_files, mode='val')


# Давайте посмотрим на наших героев внутри датасета.

# In[23]:


fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8, 8), \
                        sharey=True, sharex=True)
for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0,1000))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),\
                val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
    imshow(im_val.data.cpu(), \
          title=img_label,plt_ax=fig_x)


# Можете добавить ваши любимые сцены и классифицировать их. (веселые результаты можно кидать в чат)

# ### Построение нейросети
# 
# Запустить данную сеть будет вашим мини-заданием на первую неделю, чтобы было проще участвовать в соревновании.
# 
# Данная архитектура будет очень простой и нужна для того, чтобы установить базовое понимание и получить простенький сабмит на Kaggle
# 
# <!-- Здесь вам предлагается дописать сверточную сеть глубины 4/5.  -->
# 
# *Описание слоев*:
# 
# 
# 
# 1. размерность входа: 3x224x224 
# 2.размерности после слоя:  8x111x111
# 3. 16x54x54
# 4. 32x26x26
# 5. 64x12x12
# 6. выход: 96x5x5
# 

# In[ ]:


# Очень простая сеть
class SimpleCnn(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(96 * 5 * 5, n_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        logits = self.out(x)
        return logits


# In[ ]:


import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


# In[ ]:


def fit_epoch(model, train_loader, criterion, optimizer):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc


# In[ ]:


def eval_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size
    return val_loss, val_acc


# In[ ]:


def train(train_files, val_files, model, epochs, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        opt = torch.optim.Adam(model.parameters())
        criterion = AsymmetricLossOptimized() #nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt)
            print("loss", train_loss)

            val_loss, val_acc = eval_epoch(model, val_loader, criterion)
            history.append((train_loss, train_acc, val_loss, val_acc))

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss,\
                                           v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))

    return history


# In[ ]:


def predict(model, test_loader):
    with torch.no_grad():
        logits = []

        for inputs in test_loader:
            inputs = inputs.to(DEVICE)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)

    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs


# In[ ]:


n_classes = len(np.unique(train_val_labels))
simple_cnn = SimpleCnn(n_classes).to(DEVICE)
print("we will classify :{}".format(n_classes))
print(simple_cnn)


# In[ ]:


from torchvision import models
resnet50 = models.resnet50(pretrained = True).to(DEVICE)


# уже натренировал модель на куче картинок, вкратце, поскольку вы видим большой мисбаланс, была натренирована обычная модель(ниже), проверены ошибки которые она делает по классам и затем...взяты ролики из ютуба (best moments of Willie, Simpsons) порезаны по картинкам ffmpeg , поиск по хэшу и некоторая ручная обработка и собраны картинки для недостающих персонажей

# In[ ]:


# resnet50_2 = models.resnet50(pretrained = True).to(DEVICE)
resnet50_2 = torch.load('/content/drive/MyDrive/stepik/simpsons_pytorch/model.bin').to(DEVICE)
resnet50_2.eval()


# опционально

# In[ ]:


# ct = 0
# for child in resnet50.children():
#   ct += 1
#   if ct < 8:
#     for param in child.parameters():
#         param.requires_grad = False   
# resnet50.fc = nn.Linear(2048, 42).to(DEVICE) 


# Запустим обучение сети.

# In[ ]:


if val_dataset is None:
    val_dataset = SimpsonsDataset(val_files, mode='val')

train_dataset = SimpsonsDataset(train_files, mode='train')


# In[ ]:


history = train(train_dataset, val_dataset, model=resnet50, epochs=10, batch_size=1)


# Построим кривые обучения

# In[ ]:


loss, acc, val_loss, val_acc = zip(*history)


# In[ ]:


plt.figure(figsize=(15, 9))
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# ### Ну и что теперь со всем этим делать?

# ![alt text](https://www.indiewire.com/wp-content/uploads/2014/08/the-simpsons.jpg)

# Хорошо бы понять, как сделать сабмит. 
# У нас есть сеть и методы eval у нее, которые позволяют перевести сеть в режим предсказания. Стоит понимать, что у нашей модели на последнем слое стоит softmax, которые позволяет получить вектор вероятностей  того, что объект относится к тому или иному классу. Давайте воспользуемся этим.

# In[ ]:


def predict_one_sample(model, inputs, device=DEVICE):
    """Предсказание, для одной картинки"""
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs


# In[ ]:


random_characters = int(np.random.uniform(0,1000))
ex_img, true_label = val_dataset[random_characters]
probs_im = predict_one_sample(resnet50, ex_img.unsqueeze(0))


# In[ ]:


idxs = list(map(int, np.random.uniform(0,1000, 20)))
imgs = [val_dataset[id][0].unsqueeze(0) for id in idxs]

probs_ims = predict(resnet50, imgs)


# In[ ]:


label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))


# In[ ]:


y_pred = np.argmax(probs_ims,-1)

actual_labels = [val_dataset[id][1] for id in idxs]

preds_class = [label_encoder.classes_[i] for i in y_pred]


# Обратите внимание, что метрика, которую необходимо оптимизировать в конкурсе --- f1-score. Вычислим целевую метрику на валидационной выборке.

# In[ ]:


from sklearn.metrics import f1_score

f1_score(actual_labels, y_pred, average='micro')


# In[ ]:


y_pred!=actual_labels


# Сделаем классную визуализацию,  чтобы посмотреть насколько сеть уверена в своих ответах. Можете исспользовать это, чтобы отлаживать правильность вывода.

# In[ ]:


import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(12, 12), \
                        sharey=True, sharex=True)
for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0,1000))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),\
                val_dataset.label_encoder.inverse_transform([label])[0].split('_')))



    imshow(im_val.data.cpu(), \
          title=img_label,plt_ax=fig_x)

    actual_text = "Actual : {}".format(img_label)

    fig_x.add_patch(patches.Rectangle((0, 53),86,35,color='white'))
    font0 = FontProperties()
    font = font0.copy()
    font.set_family("fantasy")
    prob_pred = predict_one_sample(resnet50, im_val.unsqueeze(0))
    predicted_proba = np.max(prob_pred)*100
    y_pred = np.argmax(prob_pred)

    predicted_label = label_encoder.classes_[y_pred]
    predicted_label = predicted_label[:len(predicted_label)//2] + '\n' + predicted_label[len(predicted_label)//2:]
    predicted_text = "{} : {:.0f}%".format(predicted_label,predicted_proba)

    fig_x.text(1, 59, predicted_text , horizontalalignment='left', fontproperties=font,
                    verticalalignment='top',fontsize=8, color='black',fontweight='bold')


# In[ ]:


errors=[]
with torch.no_grad():
    for i in range(len(val_dataset)):
      im_val, label = val_dataset[i]
      img_label = " ".join(map(lambda x: x.capitalize(),\
                    val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
      prob_pred = predict_one_sample(resnet50, im_val.unsqueeze(0))
      predicted_proba = np.max(prob_pred)*100
      y_pred = np.argmax(prob_pred)
      predicted_label = label_encoder.classes_[y_pred]
      # predicted_label = predicted_label[:len(predicted_label)//2] + '\n' + predicted_label[len(predicted_label)//2:]
      predicted_text = "{} : {:.0f}%".format(predicted_label,predicted_proba)


      if label!=y_pred:
        errors.append({"label":label, "y_pred":y_pred, "predicted_text":predicted_text, "predicted_label":predicted_label, "img_label":img_label})


# In[ ]:


torch.save(resnet50, "/content/drive/MyDrive/stepik/simpsons_pytorch/model.bin")


# In[ ]:


outs = pd.DataFrame(errors)
outs.T


# Попробуйте найти те классы, которые сеть не смогла расспознать. Изучите данную проблему, это понадобится в дальнейшем.

# ### Submit на Kaggle

# ![alt text](https://i.redd.it/nuaphfioz0211.jpg)

# In[ ]:


test_dataset = SimpsonsDataset(test_files, mode="test")
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)
probs = predict(simple_cnn, test_loader)


preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
test_filenames = [path.name for path in test_dataset.files]


# 3 модель

# In[24]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import glob2 as glob


# In[26]:


d = dict()
for f in glob.glob("/content/train/simpsons_dataset/*"):
  d[os.path.basename(f)] = len(glob.glob(f"{f}/*.jpg"))

max_len = max(d.values())


# In[27]:


def sample_images(folder, max_len):
  arr = np.array(glob.glob(f"{folder}/*.jpg"))
  return arr[np.random.choice(np.arange(len(arr)), max_len-len(arr))]


# In[28]:


import shutil
import uuid


# In[29]:


for f in glob.glob("/content/train/simpsons_dataset/*"):
  for img_f in sample_images(f, max_len):
    dst = os.path.join(os.path.dirname(img_f), str(uuid.uuid1())+".jpg")
    shutil.copy(img_f,dst)


# In[30]:


d = dict()
for f in glob.glob("/content/train/simpsons_dataset/*"):
  d[os.path.basename(f)] = len(glob.glob(f"{f}/*.jpg"))
d


# In[34]:


train_data_dir = "/content/train/simpsons_dataset"
batch_size = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224


# In[ ]:


train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2)


# In[ ]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    batch_size=batch_size,
    shuffle = True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    subset='training') 
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    batch_size=batch_size,
    shuffle = True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    subset='validation') 


# In[31]:


get_ipython().system('pip install efficientnet')


# In[32]:


import efficientnet.keras as efn


# In[35]:


efficient_net = efn.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    )


# In[ ]:


efficient_net = tf.keras.applications.MobileNetV2(input_shape=[224,224, 3], include_top=False)
    #pretrained_model = tf.keras.applications.Xception(input_shape=[*IMAGE_SIZE, 3], include_top=False)
    #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    #pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    #pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    # EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)
    #pretrained_model = efficientnet.tfkeras.EfficientNetB0(weights='imagenet', include_top=False)
efficient_net.trainable = True


# In[ ]:


model = tf.keras.Sequential([
  efficient_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(640,kernel_regularizer=keras.regularizers.l2(0.001), activation = 'relu'), ##
  # tf.keras.layers.ReLU(max_value=6),
  tf.keras.layers.Dropout(.4),
  tf.keras.layers.Dense(240,kernel_regularizer=keras.regularizers.l2(0.001), activation = 'relu'), ##
  # tf.keras.layers.ReLU(max_value=6),
  tf.keras.layers.Dropout(.2),
  tf.keras.layers.Dense(42, activation = 'softmax')])


# In[ ]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])


# In[ ]:


history = model.fit(
    train_generator,
    # steps_per_epoch=total_train // batch_size,
    epochs=20,
    validation_data=validation_generator,
    # validation_steps=total_val // batch_size
)


# In[ ]:


model.save("/content/drive/MyDrive/stepik/tf")


# In[ ]:


path_to_data = "/content/train/simpsons_dataset"
labels_name = os.listdir(path_to_data)
labels_name.sort()
index_name = [i for i in range(0,len(labels_name))]
labels_dict = dict(zip(index_name, labels_name))


# In[ ]:


def index_to_label(model, img_data_dir):
  image = tf.keras.preprocessing.image.load_img(img_data_dir,target_size = (IMG_HEIGHT, IMG_WIDTH))
  input_arr = keras.preprocessing.image.img_to_array(image)
  input_arr = tf.keras.applications.mobilenet_v2.preprocess_input(input_arr)
  input_arr = tf.reshape(input_arr , [1, 224, 224, 3])
  input = model(input_arr).numpy()
  predict = labels_dict[int(np.argmax(input, axis = 1))]
  return predict


# In[ ]:


index_to_label(model,"/content/testset/testset/img101.jpg")


# submit

# In[ ]:


path_to_data = "/content/testset/testset"
test_names = os.listdir(path_to_data)
test_names = test_names


# In[ ]:


preds = []
for image_test in test_names:
  predict = index_to_label(model,f"{path_to_data}/{image_test}")
  preds.append(predict)


# In[ ]:


submit = pd.DataFrame(columns=['Id'])
submit['Id'] = test_names
submit[f'Expected'] = preds


# In[ ]:


submit.to_csv("submit.csv", index=False)


# Итог, было сделано несколько моделей, (всего 3) и сделан простой "блендинг", даже не ансамбль(поскольку я взял pytorch и tf модели, вместе они не тренируются), - правильным результатом считался по сути голосование, в случае несовпадения всех лейблов, результат натренированного на доп картинках resnet.  
# 
# Результат обычного реснет - 92% ,удалось повысить разбавлением доп моделями

# ![image.png](data:image/png;base64,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)

# ## Приключение?
# 
# А теперь самое интересное, мы сделали простенькую сверточную сеть и смогли отправить сабмит, но получившийся скор нас явно не устраивает. Надо с этим что-то сделать. 
# 
# Несколько срочныйх улучшейни для нашей сети, которые наверняка пришли Вам в голову: 
# 
# 
# *   Учим дольше и изменяем гиперпараметры сети
# *  learning rate, batch size, нормализация картинки и вот это всё
# *   Кто же так строит нейронные сети? А где пулинги и батч нормы? Надо добавлять
# *  Ну разве Адам наше все? [adamW](https://www.fast.ai/2018/07/02/adam-weight-decay/) для практика, [статейка для любителей](https://openreview.net/pdf?id=ryQu7f-RZ) (очень хороший анализ), [наши ](https://github.com/MichaelKonobeev/adashift/) эксперименты для заинтересованных.
# 
# * Ну разве это deep learning? Вот ResNet и Inception, которые можно зафайнтьюнить под наши данные, вот это я понимаю (можно и обучить в колабе, а можно и [готовые](https://github.com/Cadene/pretrained-models.pytorch) скачать).
# 
# * Данных не очень много, можно их аугументировать и  доучититься на новом датасете ( который уже будет состоять из, как  пример аугументации, перевернутых изображений)
# 
# * Стоит подумать об ансамблях
# 
# 
# Надеюсь, что у Вас получится!
# 
# ![alt text](https://pbs.twimg.com/profile_images/798904974986113024/adcQiVdV.jpg)
# 
