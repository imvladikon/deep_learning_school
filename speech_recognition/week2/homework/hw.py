#!/usr/bin/env python
# coding: utf-8

# <center><image src="https://drive.google.com/uc?id=1n3G4TdK_u6PQHcLrxB_A0HijNdigXmUH">

# <h3 style="text-align: center;"><b>Школа глубокого обучения ФПМИ МФТИ</b></h3>
# 
# <h3 style="text-align: center;"><b>Домашнее задание. Классификация звуков</b></h3>
# 
# **Автор**: Ермекова Асель
# 
# 
# В этом задании вам предстоит решить задачу классификации звуков на основе wav файлов и использовании различных аугментаций данных.
# 
# Есть две части этого домашнего задания.
# 
# ### 1 Часть. Отправить ваши предсказания в Stepik.
# Результат вашей лучшей модели будет оцениваться на тестовой выборке по метрике Accuracy. Эта часть оценивается до 5 баллов.
# 
# 1) $1.00 \geqslant score \geqslant 0.75$ --- 5 баллов
# 
# 2) $0.75 > score \geqslant 0.70$ --- 4 балла
# 
# 3) $0.70 > score \geqslant 0.60$ --- 3 балла
# 
# 4) $0.60 > score \geqslant 0.50$ --- 2 балла
# 
# 5) $0.50 > score \geqslant 0.25$ --- 1 балл
# 
# 6) $0.25 > score$ --- 0 баллов
# 
# Для этого мы предварительно разделили данные в задании на три части.
# 
# 1. `train.csv`. На этом наборе данных вам необходимо создать и обучить модель.
# 2. `valid.csv`. На этом наборе данных вы можете валидировать вашу модель.
# 3. `test.csv`. Предсказания для этого набора необходимо записать в файл `submission.csv` и сдать в соответствующий шаг на Stepik. Количество попыток ограничено до 100 штук. В конце ноутбука есть пример оформления файла посылки.
# 
# ### 2 Часть. Сделать полноценный отчет о вашей работе (5 баллов).
# Опишите итеративный процесс улучшения метрики:
# * как вы обработали данные, какие аугментации добавляли, что сработало, а что нет.
# * какие архитектуры модели попробовали и какие результаты получились.
# 
# В этом пункте вам необходимо отправить файл в формате .ipynb на Stepik --- для этого в домашнем задании есть отдельный шаг. Этот пункт оценивается до 5 баллов.
# 
# ### Peer-review
# Вторая часть будет проверяться в формате peer-review, т.е. вашу посылку на Stepik будут проверять 3 других студента, и медианное значение их оценок будет выставлено. Чтобы получить баллы, вам также нужно будет проверить трех других учеников. Это станет доступно после того, как вы сдадите задание сами.
# 
# 
# ### Несколько замечаний по выполнению работы
# * Во всех пунктах указания это минимальный набор вещей, которые стоит сделать. Если вы можете сделать какой-то шаг лучше или добавить что-то свое --- дерзайте!
# * Пожалуйста, перед сдачей ноутбука убедитесь, что работа чистая и понятная. Это значительно облегчит проверку и повысит ваши ожидаемые баллы.
# * Если у вас будут проблемы с решением или хочется совета, то пишите в наш чат в телеграме.
# 

# # **Environmental Sound Classification**

# ## **Task Overview**
# 
# В этом домашнем задании вам предстоит работать с датасетом различных звуков окружающей среды (собака, дождь, плач ребёнка и т. д.).

# ### **Part 1: Create Dataset**

# Первым делом давайте скачаем датасет и прилагающие csv файлы с метками класса.

# In[ ]:


get_ipython().system('gdown 1TQa-tOX1b8QxuXBcrYrTveVAwfw1XBPO # sound_classification_dataset.zip')
get_ipython().system('gdown 1BvUhnTeOvik0NeuJtMrfr7LXpHCU1DUT # train.csv')
get_ipython().system('gdown 1my0RPDQdTxvCGmnZei06tiXgKko3R4o4 # valid.csv')
get_ipython().system('gdown 1Z6BG52Tmyjxhen7DqvO59Rlz-2pAg7ks # test.csv')


# Разархивируйте zip файл, где содержатся wav файлы датасета.

# In[ ]:


get_ipython().system('unzip /content/sound_classification_dataset.zip')


# In[ ]:


import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv")
test_df = pd.read_csv("test.csv")


# Для этого задания при создании датасета вам нужно сделать обработку аудио данных следующим образом:
# * **Sample rate --> 16000**: ресэмплируйте оригинальный `sample_rate` в `sample_rate = 16000`
# * **Stereo --> Mono**: преобразуйте многоканальное аудио в моноканальное
# * **Length = X secs:** чтобы суметь создать батч, вам необходимо, чтобы длина всех ваших аудиозаписей была одинаковой, поэтому вам нужно зафиксировать длину всех аудиозаписей, и если аудио меньше заданной длины, то сделайте паддинг, если больше, обрежьте аудио до заданной длины.
# 
# * **Audio Augmentation:** используйте разные аугментации. Вы можете воспользоваться библиотеками:
#   * [torchaudio.transforms](https://docs.pytorch.org/audio/main/transforms.html)
#   * [torch_audiomentations](https://github.com/iver56/torch-audiomentations)
# 
# **ВАЖНО**: в этом домашнем задании вам нельзя переводить `wav` в мелспектрограммы.
# 
# Внизу для удобства предоставлен псевдокод, который можно заполнить необходимыми функциями, но вы можете видоизменять его как вам будет удобно.

# In[ ]:


import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import os

class SimpleAudioDataset(Dataset):
    """A dataset to load, preprocess, and augment audio files."""

    def __init__(self, ..., self.do_augmentation=False):
        # here is your code

        pass

    def __len__(self):

        return # here is your code

    def __getitem__(self, index):
        # 1. Get the file path and label
        audio_file_path = # here is your code
        label = # here is your code

        # 2. LOAD: Load the raw audio file
        # here is your code

        # 3. PREPROCESS: Apply the preprocessing steps
        signal = self._resample(signal, sample_rate) # Resample to the sample rate 16000
        signal = self._stereo_to_mono(signal) # Converts (channels, samples) -> (samples,)
        signal = self._cut_or_pad(signal) # State fixed length

        # 4. AUGMENT: Apply augmentations only if training
        if self.do_augmentation:
            signal = self._augmentation(signal)

        signal = signal.squeeze(1)

        # 5. RETURN: We now have a clean, standardized waveform and its label
        return signal, label

    # --- The Core Preprocessing Functions ---
    def _resample(self, signal, original_sr):
        # here is your code

        return signal

    def _stereo_to_mono(self, signal):
        # here is your code

        return signal

    def _cut_or_pad(self, signal):
        # here is your code

        return signal

    def _augmentation(self, signal):
        # here is your code

        return signal


# In[ ]:


train_dataset = SimpleAudioDataset(train_df,...)
valid_dataset = SimpleAudioDataset(valid_df, ...)


# ### **Part 2: Building a Model that Learns from Waveforms**

# В этом разделе вам нужно написать архитектуру по вашему выбору, которая будет решать задачу классификации на 5 классов.

# In[ ]:


import torch.nn as nn

class SoundClassificatonModel(nn.Module):
    """A simple model that processes raw waveforms."""

    def __init__(self, input_size=16000, num_classes=5):
        super().__init__()
        # here is your code


    def forward(self, x):
        # here is your code
        x = ...
        return x


# ### **Part 3: Training and Evaluation**

# В этом разделе вам нужно написать код тренировки и запустить саму тренировку и вывести лучшие значения метрики качества на train и valid данных. Для вашего удобства написана функция отображения значений лоссов и метрики accuracy.

# In[ ]:


def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
    """
    Plot training and validation metrics
    """
    epochs = range(1, len(train_losses) + 1)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accuracies, 'r-', label='Valid Accuracy', linewidth=2)
    ax2.set_title('Training and Valid Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# In[ ]:


from torch.utils.data import DataLoader
from IPython.display import clear_output


# Initialize datasets & dataloaders
train_data = # here is your code
valid_data = # here is your code

train_loader = # here is your code
valid_loader = # here is your code
test_loader = # here is your code

# Model, Loss, Optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SoundClassificatonModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

n_epochs = # here is your code
for epoch in range(n_epochs):

    # Train
    model.train()

    for signals, labels in train_loader:

        # load data to device
        signals, labels = signals.to(device), labels.to(device)

        # Forward pass
        predictions = model(signals)
        train_loss = criterion(predictions, labels)
        train_accuracy = # here is your code

        # Backward pass
        ...# here is your code

    # Evaluation
    model.eval()
    with torch.no_grad():
        for signals, labels in valid_loader:
            # load data to device
            signals, labels = signals.to(device), labels.to(device)

            # Forward pass
            predictions = model(signals)
            valid_loss = criterion(predictions, labels)
            valid_accuracy = # here is your code

    # Calculate average test loss and accuracy for this epoch
    epoch_train_loss = # here is your code
    epoch_train_acc = # here is your code

    epoch_valid_loss = # here is your code
    epoch_valid_acc = # here is your code

    # Store metrics
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)
    valid_losses.append(epoch_valid_loss)
    valid_accuracies.append(epoch_valid_acc)

    plot_metrics(train_losses, train_accuracies, valid_losses, valid_accuracies)
    clear_output(wait=True)


# In[ ]:


print("Train Accuracy = ", ...)
print("Valid Accuracy = ", ...)


# ### **Part 4. Test Demo for ESC-50**

# Для вашего удобства предоставляется код для тестирования модели и отрисовки формы сигналов, прогноза и топ-5 наиболее вероятных классов.

# In[ ]:


class ESC50TestDemo:
    def __init__(self, model, test_dataset, device):
        self.model = model
        self.test_dataset = test_dataset
        self.device = device
        self.classes = test_dataset.classes
        self.model.eval()  # Set to evaluation mode

    def predict_audio(self, signal):
        """Predict class for a single audio signal"""
        with torch.no_grad():
            signal = signal.unsqueeze(0).to(self.device)  # Add batch dimension
            outputs = self.model(signal)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]

    def run_interactive_demo(self, num_examples=1):
        """Run interactive demo with random test examples"""
        print("ESC-50 Audio Classification Demo!")
        print("=" * 60)

        # Get random test examples
        indices = np.random.choice(len(self.test_dataset), num_examples, replace=False)

        for i, idx in enumerate(indices):
            # Load audio and true label
            signal, true_label = self.test_dataset[idx]
            true_class = self.classes[true_label]

            # Get prediction
            predicted_idx, confidence, all_probs = self.predict_audio(signal)
            predicted_class = self.classes[predicted_idx]

            # Clear previous output
            # clear_output(wait=True)

            # Create plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

            # Plot waveform
            ax1.plot(signal.squeeze().numpy())
            ax1.set_title(f'Audio Waveform - Example {i+1}/{num_examples}')
            ax1.set_xlabel('Samples')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)

            # Plot prediction info
            colors = ['lightcoral', 'lightgreen']
            correct = predicted_class == true_class
            ax2.barh([0, 1], [confidence * 100, (1-confidence) * 100],
                     color=colors[correct], alpha=0.7)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels([f'Predicted: {predicted_class}',
                               f'True: {true_class}'])
            ax2.set_xlabel('Confidence (%)')
            ax2.set_title(f'Prediction ({"✓ Correct" if correct else "✗ Wrong"})')

            # Plot top-5 predictions
            top5_indices = np.argsort(all_probs)[-5:][::-1]
            top5_classes = [self.classes[idx] for idx in top5_indices]
            top5_probs = all_probs[top5_indices]

            colors = ['lightgreen' if cls == true_class else 'lightcoral' for cls in top5_classes]
            ax3.barh(range(5), top5_probs * 100, color=colors, alpha=0.7)
            ax3.set_yticks(range(5))
            ax3.set_yticklabels(top5_classes)
            ax3.set_xlabel('Probability (%)')
            ax3.set_title('Top-5 Predictions')
            ax3.invert_yaxis()  # Highest probability at top

            plt.tight_layout()
            plt.show()

            # Display audio player
            print(f"Playing: {true_class}")
            display(Audio(signal.squeeze().numpy(), rate=16000))

            print(f"Prediction: {predicted_class} ({confidence:.2%})")
            print(f"True label: {true_class}")
            print(f"Correct: {correct}")
            print("=" * 60)


    def evaluate_test_set(self):
        """Evaluate on entire test set"""
        test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)

                total += target.size(0)
                correct += (predicted == target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        accuracy = 100 * correct / total
        print(f"ESC-50 Test Set Evaluation (Fold 5):")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2f}%")

        return accuracy, all_predictions, all_labels

# Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create demo
demo = ESC50TestDemo(model, valid_dataset, device)

# Run interactive demo
demo.run_interactive_demo(num_examples=5)

# Evaluate on entire test set
test_accuracy, predictions, true_labels = demo.evaluate_test_set()


# ### **Create submission to Stepik**

# Вам нужно:
# * **1 шаг.** сделать предсказания для `test.csv` при помощи лучшей модели
# * **2 шаг.** создать `submission.csv` файл с колонкой `category`, положить туда свои предсказания и сохранить файл.

# In[ ]:


y_test_pred = # here are your predictions


# In[ ]:


submission = pd.read_csv("/content/test.csv")
submission['category'] = y_test_pred
submission.to_csv("/content/submission.csv", index=False)


# ### **Report**

# Опишите ваш путь экспериментов и что вы сделали, чтобы получить наилучшую модель.
