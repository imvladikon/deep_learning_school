#!/usr/bin/env python
# coding: utf-8

# <p style="align: center;"><img align=center src="https://drive.google.com/uc?export=view&id=1I8kDikouqpH4hf7JBiSYAeNT2IO52T-T" width=600 height=480/></p>
# <h3 style="text-align: center;"><b>Школа глубокого обучения ФПМИ МФТИ</b></h3>
# 
# <h3 style="text-align: center;"><b>Домашнее задание. Весна 2021</b></h3>
# 
# # Generative adversarial networks
# 

# В этом домашнем задании вы обучите GAN генерировать лица людей и посмотрите на то, как можно оценивать качество генерации

# In[ ]:


get_ipython().system('pip install celluloid -qq')


# In[ ]:


import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
import cv2
from tqdm.notebook import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='darkgrid', font_scale=1.2)


# ## Часть 1. Подготовка данных (1 балл)

# В качестве обучающей выборки возьмем часть датасета [Flickr Faces](https://github.com/NVlabs/ffhq-dataset), который содержит изображения лиц людей в высоком разрешении (1024х1024). Оригинальный датасет очень большой, поэтому мы возьмем его часть. Скачать датасет можно [здесь](https://drive.google.com/file/d/1KWPc4Pa7u2TWekUvNu9rTSO0U2eOlZA9/view?usp=sharing)

# Давайте загрузим наши изображения. Напишите функцию, которая строит DataLoader для изображений, при этом меняя их размер до нужного значения

# In[ ]:


DATA_DIR = 'images/'


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!mkdir -p {DATA_DIR}\n!cp /content/drive/MyDrive/edu/faces_dataset_small.zip .\n!unzip faces_dataset_small.zip -d {DATA_DIR}\n!rm -rf images/__MACOSX\n')


# In[ ]:


def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


# In[ ]:


def get_dataloader(image_size, batch_size, is_test=False):
  """
  Builds dataloader for training data.
  Use tt.Compose and tt.Resize for transformations
  :param image_size: height and wdith of the image
  :param batch_size: batch_size of the dataloader
  :returns: DataLoader object 
  """
  # TODO: resize images, convert them to tensors and build dataloader
  ds = ImageFolder(DATA_DIR, transform=tt.Compose([
    tt.Resize(image_size),
    tt.CenterCrop(image_size),
    tt.ToTensor(),
    tt.Normalize(*stats)]))
  dl = DataLoader(ds, batch_size, shuffle=not is_test, num_workers=2, pin_memory=True)
  return dl


# In[ ]:


image_size = 64
batch_size = 32
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
train_dl = get_dataloader(image_size, batch_size)


# In[ ]:


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


# In[ ]:


show_batch(train_dl)


# In[ ]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device


# In[ ]:


train_dl = DeviceDataLoader(train_dl, device)


# ## Часть 2. Построение и обучение модели (2 балла)

# Сконструируйте генератор и дискриминатор. Помните, что:
# * дискриминатор принимает на вход изображение (тензор размера `3 x image_size x image_size`) и выдает вероятность того, что изображение настоящее (тензор размера 1)
# 
# * генератор принимает на вход тензор шумов размера `latent_size x 1 x 1` и генерирует изображение размера `3 x image_size x image_size`

# In[ ]:


# на основе семинара

discriminator = nn.Sequential(
    # in: 3 x image_size x image_size

    nn.Conv2d(3, image_size, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(image_size),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid())


# In[ ]:


discriminator = to_device(discriminator, device)


# In[ ]:


latent_size = 128

generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(image_size, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x image_size x image_size
)


# Перейдем теперь к обучению нашего GANа. Алгоритм обучения следующий:
# 1. Учим дискриминатор:
#   * берем реальные изображения и присваиваем им метку 1
#   * генерируем изображения генератором и присваиваем им метку 0
#   * обучаем классификатор на два класса
# 
# 2. Учим генератор:
#   * генерируем изображения генератором и присваиваем им метку 0
#   * предсказываем дискриминаторором, реальное это изображение или нет
# 
# 
# В качестве функции потерь берем бинарную кросс-энтропию

# In[ ]:


lr = 0.0001

model = {
    "discriminator": discriminator.to(device),
    "generator": generator.to(device)
}

criterion = {
    "discriminator": nn.BCELoss(),
    "generator": nn.BCELoss()
}


# In[ ]:


def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


# In[ ]:


fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)


# In[ ]:


from IPython.display import clear_output
import os
try:
  from celluloid import Camera
except:
  pass


# In[ ]:


sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)


# In[ ]:


def fit(model, criterion, epochs, lr, start_idx=1, make_animation=True):
  # build optimizers and train your GAN
  model["discriminator"].train()
  model["generator"].train()
  torch.cuda.empty_cache()

  # Losses & scores
  losses_g = []
  losses_d = []
  real_scores = []
  fake_scores = []

  # Create optimizers
  optimizer = {
      "discriminator": torch.optim.Adam(model["discriminator"].parameters(), 
                                        lr=lr, betas=(0.5, 0.999)),
      "generator": torch.optim.Adam(model["generator"].parameters(),
                                    lr=lr, betas=(0.5, 0.999))
  }
  fig, ax = plt.subplots(figsize=(8, 8))

  if make_animation:
    camera = Camera(fig)

  for epoch in range(epochs):
      loss_d_per_epoch = []
      loss_g_per_epoch = []
      real_score_per_epoch = []
      fake_score_per_epoch = []
      for real_images, _ in tqdm(train_dl):
          # Train discriminator
          # Clear discriminator gradients
          optimizer["discriminator"].zero_grad()

          # Pass real images through discriminator
          real_preds = model["discriminator"](real_images)
          real_targets = torch.ones(real_images.size(0), 1, device=device)
          real_loss = criterion["discriminator"](real_preds, real_targets)
          cur_real_score = torch.mean(real_preds).item()

          # Generate fake images
          latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
          fake_images = model["generator"](latent)

          # Pass fake images through discriminator
          fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
          fake_preds = model["discriminator"](fake_images)
          fake_loss = criterion["discriminator"](fake_preds, fake_targets)
          cur_fake_score = torch.mean(fake_preds).item()

          real_score_per_epoch.append(cur_real_score)
          fake_score_per_epoch.append(cur_fake_score)

          # Update discriminator weights
          loss_d = real_loss + fake_loss
          loss_d.backward()
          optimizer["discriminator"].step()
          loss_d_per_epoch.append(loss_d.item())


          # Train generator
          # Clear generator gradients
          optimizer["generator"].zero_grad()

          # Generate fake images
          latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
          fake_images = model["generator"](latent)

          # Try to fool the discriminator
          preds = model["discriminator"](fake_images)
          targets = torch.ones(batch_size, 1, device=device)
          loss_g = criterion["generator"](preds, targets)

          # Update generator weights
          loss_g.backward()
          optimizer["generator"].step()
          loss_g_per_epoch.append(loss_g.item())

      # Record losses & scores
      losses_g.append(np.mean(loss_g_per_epoch))
      losses_d.append(np.mean(loss_d_per_epoch))
      real_scores.append(np.mean(real_score_per_epoch))
      fake_scores.append(np.mean(fake_score_per_epoch))

      clear_output(wait=True)

      # Log losses & scores (last batch)
      print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
          epoch+1, epochs, 
          losses_g[-1], losses_d[-1], real_scores[-1], fake_scores[-1]))


      fake_images = model["generator"](fixed_latent)
      nmax=4
      images = fake_images.detach().cpu()
      ax.set_xticks([]); ax.set_yticks([])
      ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
      plt.show();
      if make_animation:
        camera.snap()

      # Save generated images
      if epoch == epochs - 1:
        save_samples(epoch+start_idx, fixed_latent, show=False)

  if make_animation:
    animation = camera.animate()
    animation.save('animation.mp4')  

  return losses_g, losses_d, real_scores, fake_scores


# In[ ]:


fake_images = model["generator"](fixed_latent)
show_images(fake_images.detach().cpu(), 4)


# In[ ]:


epochs = 40
history = fit(model, criterion, epochs, lr)


# итак, полученное видео в итоге залито на ютуб, можно посмотреть, как там оно тренировалось
# ![image.png](data:image/png;base64,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)

# In[1]:


get_ipython().run_cell_magic('HTML', '', '<iframe width="480" height="382" src="https://www.youtube.com/embed/3qFrOVPuSRU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\n')


# Постройте графики лосса для генератора и дискриминатора. Что вы можете сказать про эти графики?

# In[ ]:


losses_g, losses_d, real_scores, fake_scores = history


# In[ ]:


plt.figure(figsize=(15, 6))
plt.plot(losses_d, '-')
plt.plot(losses_g, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses');


# в принципе логично, задача дискриминатора на бинарным лоссом она несколько проще сходится, логично получить низкий лосс, видно что можно было тренировать дольше чем 40 эпох (идеально будет когда у дискриминатора лосс будет повышаться). в случае дискриминатора вначале идет рандомный шум и видно до 5 эпохи блуждания, но потом по мере генерации картинок более высокого качества, лосс начинает падать 

# In[ ]:


plt.figure(figsize=(15, 6))

plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real', 'Fake'])
plt.title('Scores');


# ## Часть 3. Генерация изображений (1 балл)

# Теперь давайте оценим качество получившихся изображений. Напишите функцию, которая выводит изображения, сгенерированные нашим генератором

# In[ ]:


with torch.no_grad():
  n_images = 4
  fixed_latent = torch.randn(n_images, latent_size, 1, 1, device=device)
  fake_images = model["generator"](fixed_latent)
  show_images(fake_images.detach().cpu(), 4)


# Как вам качество получившихся изображений?

# ## Часть 4. Leave-one-out-1-NN classifier accuracy (6 баллов)

# ### 4.1. Подсчет accuracy (4 балла)

# Не всегда бывает удобно оценивать качество сгенерированных картинок глазами. В качестве альтернативы вам предлагается реализовать следующий подход:
#   * Сгенерировать столько же фейковых изображений, сколько есть настоящих в обучающей выборке. Присвоить фейковым метку класса 0, настоящим – 1.
#   * Построить leave-one-out оценку: обучить 1NN Classifier (`sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)`) предсказывать класс на всех объектах, кроме одного, проверить качество (accuracy) на оставшемся объекте. В этом вам поможет `sklearn.model_selection.LeaveOneOut`

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from sklearn.metrics import *
import glob2 as glob
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict


# In[ ]:


get_ipython().system('gdown https://drive.google.com/uc?id=1xWuWDBRX_AffMQymquGnyybPbJXqW6V_')
get_ipython().system('gdown https://drive.google.com/uc?id=1--hw6lYVkgJKjlC8VZegJNww15rNxbxG')


# In[ ]:


if os.path.exists("generator.pt"):
  model["generator"].load_state_dict(torch.load("generator.pt"))
if os.path.exists("discriminator.pt"):
  model["discriminator"].load_state_dict(torch.load("discriminator.pt"))


# In[ ]:


import copy
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# может не самая удачная идея, но давайте использовать наш же дискриминатор в виде feature_extractor

# In[ ]:


feature_extractor = copy.deepcopy(model["discriminator"])


# In[ ]:


# Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
feature_extractor[12] = nn.Conv2d(512, 100, kernel_size=(4, 4), stride=(1, 1), bias=False).to(device)
# sigmoid
feature_extractor[14] = Identity()


# In[ ]:


feature_extractor


# In[ ]:


GENERATE_DATA = False


# In[ ]:


get_ipython().run_cell_magic('capture', '', '\nimport uuid\n\ndef generate_samples(len_samples, dest_dir):\n    while len_samples!=0:\n      latent_tensors = torch.randn(64, latent_size, 1, 1, device=device)\n      fake_images = generator(latent_tensors)\n      fake_fname_template = \'{0}.png\'\n      for index in range(fake_images.shape[0]):\n        if len_samples==0:break\n        fake_fname = fake_fname_template.format(str(uuid.uuid4()))\n        fake_image = fake_images[index]\n        save_image(denorm(fake_image), os.path.join(dest_dir, fake_fname))\n        len_samples-=1\n\nif GENERATE_DATA:\n  !mkdir -p dataset_faces/fake\n  generate_samples(len(train_dl.dl.dataset), dest_dir="dataset_faces/fake")\n  !mkdir -p dataset_faces/real\n\n  dest_dir = "dataset_faces/real" \n  for i in range(len(train_dl.dl.dataset)):\n    real_image = train_dl.dl.dataset[i][0]\n    save_image(denorm(real_image), os.path.join(dest_dir, \'{0}.png\'.format(str(uuid.uuid4()))))\nelse:\n  !gdown https://drive.google.com/uc?id=1-39Uf4wWeVh8cP1gO7t1Fhgg-YN_X7Oa\n  !unzip dataset_faces.zip\n')


# In[ ]:


img_count = len(glob.glob("dataset_faces/*/*.png"))
img_count


# In[ ]:


X = np.zeros((img_count, 100))
y = np.zeros((img_count))


# In[ ]:


def get_dataloader(image_size, batch_size, data_dir):
  ds = ImageFolder(data_dir, transform=tt.Compose([
    tt.Resize(image_size),
    tt.CenterCrop(image_size),
    tt.ToTensor(),
    tt.Normalize(*stats)]))
  dl = DataLoader(ds, batch_size, shuffle=False)
  return dl


# In[ ]:


# jupyter "bug"
get_ipython().system('rm -rf dataset_faces/.ipynb_checkpoints')


# In[ ]:


image_dl = get_dataloader(image_size, batch_size=1, data_dir="dataset_faces")


# In[ ]:


image_dl.dataset.class_to_idx


# In[ ]:


i = 0
for batch_x, batch_labels in image_dl:
  X[i:i+len(batch_x)] = feature_extractor(batch_x.to(device)).detach().cpu().numpy()
  y[i:i+len(batch_x)] = batch_labels.detach().cpu().numpy()
  i+=len(batch_x)


# In[ ]:


loo = LeaveOneOut()
loo.get_n_splits(X)


# In[ ]:


actual = []
predicted = []

for train_index, test_index in tqdm(loo.split(X), total=loo.get_n_splits(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifier = KNeighborsClassifier(n_neighbors=1)  
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    actual.append(y_test[0])
    predicted.append(y_pred[0])


# In[ ]:


print(classification_report(actual, predicted))


# Что вы можете сказать о получившемся результате? Какой accuracy мы хотели бы получить и почему?

# > ну результат по классификации хороший, а значит плохой) мы хотели бы получать картинки, плохо отличающиеся от реальных, а следовательно ожидали бы accuracy в районе рандома ~50% . с другой стороны по первому ближайшему соседу классифицировать, вероятно находятся пара точек одного класса в одной плоскости в большинстве случаев близко, что говорит о несовршенстве нашего GAN (для генерации мы "батчами" брали новый latent vector)

# ### 4.2. Визуализация распределений (2 балла)

# Давайте посмотрим на то, насколько похожи распределения настоящих и фейковых изображений. Для этого воспользуйтесь методом, снижающим размерность (к примеру, TSNE) и изобразите на графике разным цветом точки, соответствующие реальным и сгенерированным изображенияи

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install umap-learn\n!pip install umap-learn[plot]\n')


# In[ ]:


import umap.plot

mapper = umap.UMAP(n_neighbors=2).fit(X)
umap.plot.points(mapper, labels=y, color_key_cmap='Paired', background='black')


# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


X_2d = TSNE(n_components=2).fit_transform(X)
colors = {0: 'r', 1: 'g'}
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y, palette=colors)


# Прокомментируйте получившийся результат:

# В целом результат не так уж и плох, по TSNE и UMAP мы видим что классы как миним линейно не разделимы, уже хорошо. по UMAP хорошо видно, что фейковые картинки лежат близко и не выделяются в отдельный кластер
