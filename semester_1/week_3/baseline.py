#!/usr/bin/env python
# coding: utf-8

# <p style="align: center;"><img align=center src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" width=500 height=450/></p>
# 
# <h3 style="text-align: center;"><b>Школа глубокого обучения ФПМИ МФТИ</b></h3>
# 
# <h3 style="text-align: center;"><b>Домашнее задание. Продвинутый поток. Осень 2020</b></h3>
# 
# Это домашнее задание будет посвящено полноценному решению задачи машинного обучения.

# Есть две части этого домашнего задания: 
# * Сделать полноценный отчет о вашей работе: как вы обработали данные, какие модели попробовали и какие результаты получились (максимум 10 баллов). За каждую выполненную часть будет начислено определенное количество баллов.
# * Лучшее решение отправить в соревнование на [kaggle](https://www.kaggle.com/t/f50bc21dbe0e42dabe5e32a21f2e5235) (максимум 5 баллов). За прохождение определенного порогов будут начисляться баллы.
# 
# 
# **Обе части будут проверяться в формате peer-review. Т.е. вашу посылку на степик будут проверять несколько других студентов и аггрегация их оценок будет выставлена. В то же время вам тоже нужно будет проверить несколько других учеников.**
# 
# **Пожалуйста, делайте свою работу чистой и понятной, чтобы облегчить проверку. Если у вас будут проблемы с решением или хочется совета, то пишите в наш чат в телеграме или в лс @runfme. Если вы захотите проаппелировать оценку, то пипшите в лс @runfme.**
# 
# **Во всех пунктах указания это минимальный набор вещей, которые стоит сделать. Если вы можете сделать какой-то шаг лучше или добавить что-то свое - дерзайте!**

# # Как проверять?
# 
# Ставьте полный балл, если выполнены все рекомендации или сделано что-то более интересное и сложное. За каждый отсустствующий пункт из рекомендация снижайте 1 балл.

# # Метрика. 
# 
# Перед решением любой задачи важно понимать, как будет оцениваться ваше решение. В данном случае мы используем стандартную для задачи классификации метрику ROC-AUC. Ее можно вычислить используя только предсказанные вероятности и истинные классы без конкретного порога классификации + она раотает даже если классы в данных сильно несбалансированны (примеров одного класса в десятки раз больше примеров длугого). Именно поэтому она очень удобна для соревнований.
# 
# Посчитать ее легко:
# 

# # Первая часть. Исследование.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%%bash\npip install catboost\npip install category_encoders\npip install sklearn-pandas\npip install imblearn\n')


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


get_ipython().run_cell_magic('capture', '', '%%bash\n\npip install kaggle\ncp -r /content/drive/MyDrive/colab_settings/.kaggle /root/\nkaggle competitions download -c advanced-dls-spring-2021\n')


# In[4]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LogisticRegressionCV
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline, Pipeline, make_union
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn_pandas import DataFrameMapper
from sklearn.utils import shuffle, class_weight
from sklearn.feature_selection import *
from catboost import CatBoostClassifier, Pool, cv as catboost_cv
import catboost as cb
from sortedcontainers import SortedList
import copy
import collections
from itertools import product,chain
from category_encoders import CatBoostEncoder
import random
random_state = 42
random.seed(42)
np.random.seed(42)
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# In[5]:


###helpers functions

def display_classification_report(y_true, y_pred):
    display(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

def plot_roc(y_test, preds, ax=None, label='model'):
    with plt.style.context('seaborn-whitegrid'):
        if not ax: fig, ax = plt.subplots(1, 1)
        fpr, tpr, thresholds = roc_curve(y_test, preds)
        ax.plot([0, 1], [0, 1],'r--')
        ax.plot(fpr, tpr, lw=2, label=label)
        ax.legend(loc='lower right')
        ax.set_title(
             'ROC curve\n'
            f""" AP: {average_precision_score(
                y_test, preds, pos_label=1
            ):.2} | """
            f'AUC: {auc(fpr, tpr):.2}')
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.annotate(f'AUC: {auc(fpr, tpr):.2}', xy=(.43, .025))
        ax.legend()
        ax.grid()
        return ax


def plot_pr(y_test, preds, ax=None, label='model'):
    with plt.style.context('seaborn-whitegrid'):
        precision, recall, thresholds = precision_recall_curve(y_test, preds)
        if not ax: fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0],'r--')    
        ax.plot(recall, precision, lw=2, label=label)
        ax.legend()
        ax.set_title(
            'Precision-recall curve\n'
            f""" AP: {average_precision_score(
                y_test, preds, pos_label=1
            ):.2} | """
            f'AUC: {auc(recall, precision):.2}'
        )
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid()
        return ax


#missing value ratio
def missing_values(df):
    df_nulls=pd.concat([df.dtypes, df.isna().sum(), df.isna().sum()/len(df)], axis=1)
    df_nulls.columns = ["type","count","missing_ratio"]
    df_nulls=df_nulls[df_nulls["count"]>0]
    df_nulls.sort_values(by="missing_ratio", ascending=False)
    return df_nulls

#outliers by 3 sigma rule
def outlier(data):
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [x for x in data if x < lower or x > upper]
    outliers_removed = [x for x in data if x >= lower and x <= upper]
    return len(outliers)

# full description statistics 
def describe_full(df, target_name=""):
    data_describe = df.describe().T
    df_numeric = df._get_numeric_data()
    if target_name in df.columns:
        corr_with_target=df_numeric.drop(target_name, axis=1).apply(lambda x: x.corr(df_numeric[target_name]))
        data_describe['corr_with_target']=corr_with_target
    dtype_df = df_numeric.dtypes
    data_describe['dtypes'] = dtype_df
    data_null = df_numeric.isnull().sum()/len(df) * 100
    data_describe['Missing count'] = df_numeric.isnull().sum()
    data_describe['Missing %'] = data_null
    Cardinality = df_numeric.apply(pd.Series.nunique)
    data_describe['Cardinality'] = Cardinality
    df_skew = df_numeric.skew(axis=0, skipna=True)
    data_describe['Skew'] = df_skew
    data_describe['outliers']=[outlier(df_numeric[col]) for col in df_numeric.columns]
    data_describe['kurtosis']=df_numeric.kurtosis()
    return data_describe

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
        sns.kdeplot(df.loc[df[groupby] == value][on],                     shade=True, color=color, label=value, ax=ax)
    if not title:
      title = str("Distribution of " + on + " per " + groupby + " group")

    ax.set_title(title,fontsize=10)
    ax.set_xlabel(on, fontsize=10)
    return ax 


# ## Загрузка данных (2 балла)
# 
# 1) Посмотрите на случайные строчки. 
# 
# 2) Посмотрите, есть ли в датасете незаполненные значения (nan'ы) с помощью data.isna() или data.info() и, если нужно, замените их на что-то. Будет хорошо, если вы построите табличку с количеством nan в каждой колонке.

# In[6]:


# Для вашего удобства списки с именами разных колонок

# Числовые признаки
num_cols = [
    'ClientPeriod',
    'MonthlySpending',
    'TotalSpent'
]

# Категориальные признаки
cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

feature_cols = num_cols + cat_cols
target_col = 'Churn'


# In[7]:


# data_train = pd.read_csv('train.csv')
# data_train["split"] = "train"
# data_test = pd.read_csv('test.csv')
# data_test["split"] = "test"
# data = pd.concat([data_train, data_test], axis=0)


# In[7]:


data = pd.read_csv('train.csv')


# In[8]:


data = shuffle(data, random_state=random_state)


# In[9]:


data.info()


# In[10]:


data.head()


# In[11]:


describe_full(data)


# In[12]:


pd.DataFrame({"col":data.select_dtypes(include="object").columns,
              "unique": [len(data[col].unique()) for col in data.select_dtypes(include="object").columns],
              "values": [data[col].unique() for col in data.select_dtypes(include="object").columns]})


# как мы видим в основном мы имеем дело с категориальными величинами, пропущенных значений нет, но давайте переведем object в numeric для totalspent 

# In[13]:


try:
  pd.to_numeric(data["TotalSpent"], errors="raise")
except Exception as e:
  print(e)


# In[14]:


def convert_types_data(data):
  data["TotalSpent"] = data["TotalSpent"].str.strip()
  data.loc[data["TotalSpent"] == "", "TotalSpent"] = "0.0" 
  data["TotalSpent"] = pd.to_numeric(data["TotalSpent"], errors="raise")
  return data


# In[15]:


data = convert_types_data(data)


# In[16]:


data.duplicated().sum()


# In[17]:


data[data.duplicated()]


# давайте уберем дубли

# In[18]:


data = data[~data.duplicated()]


# ## Анализ данных (3 балла)
# 
# 1) Для численных призанков постройте гистограмму (*plt.hist(...)*) или boxplot (*plt.boxplot(...)*). Для категориальных посчитайте количество каждого значения для каждого признака. Для каждой колонки надо сделать *data.value_counts()* и построить bar диаграммы *plt.bar(...)* или круговые диаграммы *plt.pie(...)* (хорошо, елси вы сможете это сделать на одном гарфике с помощью *plt.subplots(...)*). 
# 
# 2) Посмотрите на распределение целевой переменной и скажите, являются ли классы несбалансированными.
# 
# 3) (Если будет желание) Поиграйте с разными библиотеками для визуализации - *sns*, *pandas_visual_analysis*, etc.
# 
# Второй пункт очень важен, потому что существуют задачи классификации с несбалансированными классами. Например, это может значить, что в датасете намного больше примеров 0 класса. В таких случаях нужно 1) не использовать accuracy как метрику 2) использовать методы борьбы с imbalanced dataset (обычно если датасет сильно несбалансирован, т.е. класса 1 в 20 раз меньше класса 0).

# In[19]:


pd.DataFrame({"col":cat_cols,
              "unique": [len(data[col].unique()) for col in cat_cols],
              "values": [data[col].unique() for col in cat_cols]})


# In[ ]:


for col in num_cols:
  plt.hist(data[col])
  plt.title(col)
  plt.show()


# In[ ]:


sns.boxplot(data['TotalSpent'])


# In[ ]:


sns.countplot(data["Churn"])


# действительно несбалансированно, но не сказать что фатально

# наиболее важные фичи по мнению катбуст

# In[26]:


X, y = data[feature_cols], data[target_col]
X_train, X_test, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state)
train_ds = Pool(data=X_train, label=y_train, cat_features=cat_cols, feature_names=feature_cols)
test_ds = Pool(data=X_test, label=y_val, cat_features=cat_cols, feature_names=feature_cols)
full_ds = Pool(data=X, label=y, cat_features=cat_cols, feature_names=feature_cols)

classes = np.unique(y_train)
weights = class_weight.compute_class_weight('balanced', classes, y_train)
class_weights = dict(zip(classes, weights))

cb = CatBoostClassifier(verbose=0, class_weights=class_weights,task_type="GPU", 
                        devices='0:1', random_seed=random_state).fit(train_ds)
df_feature_importances = pd.DataFrame(((zip(cb.feature_names_, cb.get_feature_importance())))).rename(columns={0:"feature",1:"coeff"}).sort_values(by="coeff", ascending = False )
sns.barplot(data=df_feature_importances, x=df_feature_importances["coeff"], y=df_feature_importances["feature"])


# In[29]:


try:
  del X, y, X_train, X_val, y_train, y_val, train_ds, test_ds, full_ds
except:
  pass


# In[30]:


fig, axs = plt.subplots(4, len(cat_cols) // 4, figsize=(15,15))
axs = axs.flatten()
plt.tight_layout()
for cat_col, ax in zip(cat_cols, axs):
     data[cat_col].value_counts().plot(kind='bar', ax=ax)
     ax.legend()

plt.show()


# In[31]:


display(data["Churn"].value_counts())
100*data["Churn"].value_counts()/len(data)


# как видим классы несбалансированы, неплохо бы оверсэмплить(либо указывать веса катабусту)

# In[32]:


fig, axs = plt.subplots(4, len(cat_cols) // 4, figsize=(15,15))
axs = axs.flatten()

for cat_col, ax in zip(cat_cols, axs):
     display_group_density_plot(data, groupby = cat_col, on = target_col, \
                                           palette = sns.color_palette('Set2'), 
                                title=cat_col,
                           figsize = (10, 5), ax=ax)

plt.tight_layout()
plt.show()


# In[33]:


fig, axs = plt.subplots(3, len(num_cols) // 3, figsize=(15,15))
axs = axs.flatten()

for num_col, ax in zip(num_cols, axs):
     display_group_density_plot(data, groupby = num_col, on = target_col, \
                                           palette = sns.color_palette('Set2'), 
                                title=num_col,
                           figsize = (10, 5), ax=ax)

plt.tight_layout()
plt.show()


# In[ ]:


sns.distplot(data["TotalSpent"])
plt.axvline(0, c="r", label="")
plt.legend()


# (Дополнительно) Если вы нашли какие-то ошибки в данных или выбросы, то можете их убрать. Тут можно поэксперементировать с обработкой данных как угодно, но не за баллы.

# числовые данные плюс минус в порядке, TotalSpent можно отскалировать, категориальные не смотрел

# In[ ]:


for col in num_cols:
  sns.boxplot(data[col])
  plt.title(col)
  plt.show()


# In[ ]:


describe_full(data, target_col)


# In[ ]:


data.corr()


# ClientPeriod и TotalSpent сильно коррелируют, что логично, также MonthlySpending коррелирует с TotalSpent, можно попробовать в дальнейшем убирать сильно коррелирующие фичи

# In[20]:


encoder = CatBoostEncoder(cols=cat_cols)


# In[33]:


df = data.copy()
df[cat_cols] = encoder.fit_transform(df[cat_cols], df["Churn"])


# In[42]:


def highlight_values(s, value=0.65):
    is_values = s.abs() >= value
    return ['background-color: green' if v else '' for v in is_values]


# In[43]:


df.corr().style.apply(highlight_values)


# довольно много фич коррелирует между собой, 
# 1. TotalSpent и ClientPeriod, отмеченные ранее, 
# 2. набор категориальных фичей, связанных с интернет/кино сервисами (HasOnlineBackup, HasDeviceProtection, HasTechSupportAccess, HasOnlineTV, HasMovieSubscription), это можно также увидеть по heatmap. их либо надо как-то объединить в одно по смыслу, либо убрать лишнее, либо использовать recursive feature elimination. 
# 3. HasContractPhone и ClientPeriod
# 4. довольно неожиданно, MonthlySpending и HasInternetService, больше коррелируют чем, связанные по смыслу MonthlySpending и TotalSpent. возможно клиенты больше ориентированы на интернет, чем на что либо еще
# 

# In[45]:


sns.heatmap(df.corr());


# помимо TotalSpent vs ClientPeriod, прослеживается тенденция, чем период клиента и он больше платит, тем больше шанс его ухода. по логике вещей в подобных кейсах лучше делить клиентов по группам и исследовать каждую (а возможно и строить модель) отдельно, но из простоты, будем все же строить одну модель

# можно еще проверить значимость категории по chi square test

# In[50]:


import scipy.stats as ss
stats = []
imp_stats = []
imp_cat_cols = []
for cat_col in cat_cols:
    contingency_table = pd.crosstab(data[cat_col], data['Churn'])
    chi2_stat, p_value = ss.chi2_contingency(observed=contingency_table)[:2]
    stats.append(chi2_stat)
    if p_value < 0.01:
        imp_stats.append(chi2_stat)
        imp_cat_cols.append(cat_col)
plt.figure(figsize=(12, 6))
series = pd.Series(stats, index=cat_cols).sort_values(ascending=True)
series.plot(kind='barh')
plt.title('Chi2 statistics for categorical features');


# как видим Sex и HasPhoneService не значимы (могу сказать, забегая вперед это видно и по результатам модели, используя и убирая эти категории)

# ## Применение линейных моделей (3 балла)
# 
# 1) Обработайте данные для того, чтобы к ним можно было применить LogisticRegression. Т.е. отнормируйте числовые признаки, а категориальные закодируйте с помощью one-hot-encoding'а. 
# 
# 2) С помощью кроссвалидации или разделения на train/valid выборку протестируйте разные значения гиперпараметра C и выберите лучший (можно тестировать С=100, 10, 1, 0.1, 0.01, 0.001) по метрике ROC-AUC. 
# 
# Если вы разделяете на train/valid, то используйте LogisticRegressionCV. Он сам при вызове .fit() подберет параметр С. (не забудьте передать scroing='roc_auc', чтобы при кроссвалидации сравнивались значения этой метрики, и refit=True, чтобы при потом модель обучилась на всем датасете с лучшим параметром C). 
# 
# 
# (более сложный вариант) Если вы будете использовать кроссвалидацию, то преобразования данных и LogisticRegression нужно соединить в один Pipeline с помощью make_pipeline, как это делалось во втором семинаре. Потом pipeline надо передать в GridSearchCV. Для one-hot-encoding'a можно испльзовать комбинацию LabelEncoder + OneHotEncoder (сначала превращаем строчки в числа, а потом числа првращаем в one-hot вектора.)

# для линейной модели можно бинарные категории просто в int перевести

# In[37]:


df = data.copy()


# все же как было замечено выше, давайте выделим клиентов по группам в зависимости от ClientPeriod, поскольку мы заметили разницу в поведении

# In[39]:


def map_client_period(ClientPeriod):
    if ClientPeriod < 13:
        return "0-12"
    elif ClientPeriod < 25:
        return "12-24"
    elif ClientPeriod < 49:
        return "24-48"
    else:
        return ">48"


# In[40]:


df["ClientPeriodGroup"] = df["ClientPeriod"].apply(map_client_period)


# была идея объединить сервисы в один, но простое sum , не дало прироста метрики, было проверено

# In[ ]:


# df["HasServices"] = (df["HasOnlineBackup"].astype("category").cat.codes + df["HasDeviceProtection"].astype("category").cat.codes + df["HasTechSupportAccess"].astype("category").cat.codes + df["HasOnlineTV"].astype("category").cat.codes + df["HasMovieSubscription"].astype("category").cat.codes)


# In[ ]:


# for col in ["HasPartner",	"HasChild", "HasPhoneService", "IsBillingPaperless"]:
#   df[col] = (df[col] == "Yes").astype("int")
# df["Sex"] = (df["Sex"] == "Male").astype("int")
# df = pd.get_dummies(df, drop_first=True)


# In[45]:


df["MonthlySpending"].hist()


# In[46]:


df["MonthlySpendingGroup"] = (df["MonthlySpending"]<40).astype("int")


# In[57]:


X, y = df[set(feature_cols) - set(["Sex", "HasPhoneService"])], df[target_col]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


# In[49]:


feature_cols = [
#  'ClientPeriod',
#  'MonthlySpending',
'MonthlySpendingGroup',
 'TotalSpent',
#  'Sex',
 'IsSeniorCitizen',
 'HasPartner',
 'HasChild',
#  'HasPhoneService',
 'HasMultiplePhoneNumbers',
 'HasInternetService',
 'HasOnlineSecurityService',
 'HasOnlineBackup',
 'HasDeviceProtection',
 'HasTechSupportAccess',
 'HasOnlineTV',
 'HasMovieSubscription',
 'HasContractPhone',
 'IsBillingPaperless',
 'PaymentMethod',
#  'HasServices',
 'ClientPeriodGroup']


# In[50]:


cat_cols = [
            # 'Sex',
 'IsSeniorCitizen',
 'HasPartner',
 'HasChild',
#  'HasPhoneService',
'MonthlySpendingGroup',
 'HasMultiplePhoneNumbers',
 'HasInternetService',
 'HasOnlineSecurityService',
 'HasOnlineBackup',
 'HasDeviceProtection',
 'HasTechSupportAccess',
 'HasOnlineTV',
 'HasMovieSubscription',
 'HasContractPhone',
 'IsBillingPaperless',
 'PaymentMethod',
#  'HasServices',
 'ClientPeriodGroup']


# In[51]:


X, y = df[feature_cols], df[target_col]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


# In[87]:


scaling_mapper = DataFrameMapper([
     (["TotalSpent"], StandardScaler())
 ])
ohe_mapper = DataFrameMapper([
     (cat_cols, OneHotEncoder())
 ])

pipe = Pipeline([
    ('encoder', make_union(scaling_mapper, ohe_mapper)),                 
    ('model', LogisticRegressionCV(class_weight=class_weights, cv=StratifiedKFold(n_splits=4), Cs=np.linspace(1, 10, 10), random_state=42, refit=True, scoring="roc_auc", solver='liblinear', max_iter=500))
])
pipe.fit(X_train, y_train);


# In[88]:


y_pred = pipe.predict(X_val)
y_pred_proba = pipe.predict_proba(X_val)[:, 1]
y_train_pred_proba = pipe.predict_proba(X_val)[:, 1]

print(f'ROC-AUC LR CV: {roc_auc_score(y_val, y_pred_proba)}')


# In[89]:


fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
plot_pr(y_val, y_pred_proba, ax=ax[0],label="LogisticRegressionCV")
plot_roc(y_val, y_pred_proba, ax=ax[1],label="LogisticRegressionCV")


# один из способов закодировать категории это использовать get_dummies, либо даже вручную бинарно закодировать бинарные фичи, но дайте воспользуемся CatBoostEncoder

# In[62]:


encoder = CatBoostEncoder(cols=cat_cols)

scaling_mapper = DataFrameMapper([
     (["TotalSpent"], StandardScaler())
 ])

pipe = Pipeline([
    ("encoder", make_union(scaling_mapper, encoder)),
    ('model', LogisticRegressionCV(cv=StratifiedKFold(n_splits=4), Cs=np.linspace(1, 10, 10), random_state=42, refit=True, scoring="roc_auc", solver='liblinear', max_iter=500))
]).fit(X_train, y_train)

y_pred = pipe.predict(X_val)
y_pred_proba = pipe.predict_proba(X_val)[:, 1]
y_train_pred_proba = pipe.predict_proba(X_val)[:, 1]

print(f'ROC-AUC LR CV: {roc_auc_score(y_val, y_pred_proba)}')


# In[60]:


display_classification_report(y_val, y_pred)


# In[61]:


fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
plot_pr(y_val, y_pred_proba, ax=ax[0],label="LogisticRegressionCV")
plot_roc(y_val, y_pred_proba, ax=ax[1],label="LogisticRegressionCV")


# In[90]:


from sklearn.compose import *

preprocess = ColumnTransformer(remainder='passthrough',
                               transformers=[

                                             ('binary_data', OneHotEncoder(drop='first'), cat_cols),

                                             ('scale_data', StandardScaler(), ['TotalSpent'])
                                            ])


# In[91]:


pip_log = Pipeline(steps=[
                          ('pre_processing', preprocess),
                          ('logistic_regression', LogisticRegression(class_weight=class_weights, max_iter=1000, 
                                                                     random_state=42))
                         ]
                  )


# In[92]:


params = {'logistic_regression__C': np.linspace(0, 10, 10),
          'logistic_regression__class_weight': ['none', 'balanced'],
          'logistic_regression__solver': ['newton-cg', 'lbfgs', 'sag']}

grid = GridSearchCV(pip_log, params, cv=5, n_jobs=-1, scoring='roc_auc', verbose=1).fit(X_train, y_train)

print(f'Best score for pipeline: {grid.best_score_}')
print(f'Best pipeline: {grid.best_estimator_}')


y_pred_proba = grid.predict_proba(X_val)[:, 1]

fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
plot_pr(y_val, y_pred_proba, ax=ax[0],label="LogisticRegressionCV")
plot_roc(y_val, y_pred_proba, ax=ax[1],label="LogisticRegressionCV")


# Выпишите какое лучшее качество и с какими параметрами вам удалось получить

# были попробованы CV с one hot encoding и catboostencoder из библиотеки, лучшее качество было в первом случае 0.82

# ## Применение градиентного бустинга (2 балла)
# 
# Если вы хотите получить баллы за точный ответ, то стоит попробовать градиентный бустинг. Часто градиентный бустинг с дефолтными параметрами даст вам 80% результата за 0% усилий.
# 
# Мы будем использовать catboost, поэтому нам не надо кодировать категориальные признаки. catboost сделает это сам (в .fit() надо передать cat_features=cat_cols). А численные признаки нормировать для моделей, основанных на деревьях не нужно.
# 
# 1) Разделите выборку на train/valid. Протестируйте catboost cо стандартными параметрами.
# 
# 2) Протестируйте разные занчения параметроа количества деревьев и learning_rate'а и выберите лучшую по метрике ROC-AUC комбинацию. 
# 
# (Дополнительно) Есть некоторые сложности с тем, чтобы использовать CatBoostClassifier вместе с GridSearchCV, поэтому мы не просим использовать кроссвалидацию. Но можете попробовать)

# In[76]:


X, y = df[feature_cols], df[target_col]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state)
train_ds = Pool(data=X_train, label=y_train, cat_features=cat_cols, feature_names=feature_cols)
test_ds = Pool(data=X_val, label=y_val, cat_features=cat_cols, feature_names=feature_cols)
full_ds = Pool(data=X, label=y, cat_features=cat_cols, feature_names=feature_cols)


# In[77]:


scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()


# In[78]:


cb = CatBoostClassifier(verbose=0, scale_pos_weight=scale_pos_weight,task_type="GPU", 
                        devices='0:1', random_seed=random_state).fit(train_ds)


# In[79]:


df_feature_importances = pd.DataFrame(((zip(cb.feature_names_, cb.get_feature_importance())))).rename(columns={0:"feature",1:"coeff"}).sort_values(by="coeff", ascending = False )
sns.barplot(data=df_feature_importances, x=df_feature_importances["coeff"], y=df_feature_importances["feature"])


# In[80]:


y_pred = cb.predict(test_ds)
print(cb.score(train_ds))


# In[192]:


print(accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
display_classification_report(y_val, y_pred)

y_pred_proba = cb.predict_proba(X_val)[:, 1]
y_train_pred_proba = cb.predict_proba(X_val)[:, 1]


_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_val, y_pred_proba, ax=axs[0], label="CatBoostClassifier")
plot_roc(y_val, y_pred_proba, ax=axs[1], label="CatBoostClassifier")


# In[97]:


param_grid = {
        'learning_rate': [0.03, 0.003],
        'depth': [16],
        'l2_leaf_reg': [3, 7],
        'iterations':[1000],
        'thread_count':[12],
        'border_count':[128]
}



model = CatBoostClassifier(eval_metric='AUC:hints=skip_train~false', task_type="GPU", devices='0:1', random_seed=random_state, 
                                       thread_count=-1)
grid_search_result = model.grid_search(param_grid, 
                                       full_ds,
                                       verbose=0,
                                       partition_random_seed=random_state,
                                       search_by_train_test_split=True,
                                       train_size=0.9,
                                       plot=False)
# eval_set = [(X_val, y_val)], use_best_model=True


# In[98]:


cv_data = pd.DataFrame(grid_search_result["cv_results"])
best_value = cv_data['test-AUC-mean'].max()
best_iter = cv_data['test-AUC-mean'].values.argmax()

print('Best validation test-AUC-mean : {:.4f}±{:.4f} on step {}'.format(
    best_value,
    cv_data['test-AUC-std'][best_iter],
    best_iter)
)


# In[100]:


model = CatBoostClassifier(task_type="GPU", devices='0:1', random_seed=random_state, **grid_search_result["params"])
model.fit(train_ds, verbose = 0, eval_set = [(X_val, y_val)], use_best_model=True)
y_pred = model.predict(test_ds)
print("accuracy_score", accuracy_score(y_val, y_pred))
for i in [10, 15, 20]:
  print("roc_auc_score", roc_auc_score(y_val, model.predict_proba(test_ds, ntree_start=0, ntree_end=i)[:,1]))
print("f1_score", f1_score(y_val, y_pred))


# In[101]:


print(roc_auc_score(y_val, model.predict_proba(test_ds)[:,1]))


# In[147]:


class paramsearch:
    def __init__(self,pdict):    
        self.pdict = {}
        # if something is not passed in as a sequence, make it a sequence with 1 element
        #   don't treat strings as sequences
        for a,b in pdict.items():
            if isinstance(b, collections.Sequence) and not isinstance(b, str): self.pdict[a] = b
            else: self.pdict[a] = [b]
        # our results are a sorted list, so the best score is always the final element
        self.results = SortedList()       

    def grid_search(self,keys=None):
        # do grid search on only the keys listed. If none provided, do all
        if keys==None: keylist = self.pdict.keys()
        else: keylist = keys

        listoflists = [] # this will be list of lists of key,value pairs
        for key in keylist: listoflists.append([(key,i) for i in self.pdict[key]])
        for p in product(*listoflists):
            # do any changes to the current best parameter set
            if len(self.results)>0: template = self.results[-1][1]
            else: template = {a:b[0] for a,b in self.pdict.items()}
            # if our updates are the same as current best, don't bother
            if self.equaldict(dict(p),template): continue
            # take the current best and update just the ones to change
            yield self.overwritedict(dict(p),template)

    def equaldict(self,a,b):
        for key in a.keys(): 
            if a[key] != b[key]: return False
        return True            

    def overwritedict(self,new,old):
        old = copy.deepcopy(old)
        for key in new.keys(): old[key] = new[key]
        return old            

    # save a (score,params) pair to results. Since 'results' is a sorted list,
    #   the best score is always the final element. A small amount of noise is added
    #   because sorted lists don't like it when two scores are exactly the same    
    def register_result(self,result,params):
        self.results.add((result+np.random.randn()*1e-10,params))    

    def bestscore(self):
        return self.results[-1][0]

    def bestparam(self):
        return self.results[-1][1]


# In[151]:


def run_grid_search(train_set, test_set, y_train, y_test):


    # X, y = data[feature_cols], data[target_col]
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state)
    # full_ds = Pool(data=X, label=y, cat_features=cat_cols, feature_names=feature_cols)

    colnames = feature_cols
    category_cols = cat_cols
    cat_dims = [train_set.columns.get_loc(i) for i in category_cols[:-1]] 
    # for header in category_cols:
    #     train_set[header] = train_set[header].astype('category').cat.codes
    #     test_set[header] = test_set[header].astype('category').cat.codes

    # split labels out of data sets    
    train_label = y_train.values
    test_label = y_test.values

    # params = {'depth':[3,1,2,6,4,5,7,8,9,10],
    #           'iterations':[250,100,500,1000],
    #           'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
    #           'l2_leaf_reg':[3,1,5,10,100],
    #           'border_count':[32,5,10,20,50,100,200],
    #           'thread_count':4}

    params = {'depth':[4],
              'iterations':[500,1000, 1500],
              'learning_rate':[0.03,0.003, 0.001], 
              'l2_leaf_reg':[3],
              'border_count':[12,13,14],
              'thread_count':[4],
              'bagging_temperature':[0,1,2]}

    #           {'border_count': 12,
    #  'depth': 4,
    #  'iterations': 500,
    #  'l2_leaf_reg': 3,
    #  'learning_rate': 0.03,
    #  'thread_count': 4}

    #  {'depth': 4, 'iterations': 500, 'learning_rate': 0.03, 'l2_leaf_reg': 3, 'border_count': 10, 'thread_count': 4}
    # this function does 3-fold crossvalidation with catboostclassifier          
    def crossvaltest(params,train_set,train_label,cat_dims,n_splits=3):
        kf = KFold(n_splits=n_splits,shuffle=True) 
        res = []
        for train_index, test_index in kf.split(train_set):
            train = train_set.iloc[train_index,:]
            test = train_set.iloc[test_index,:]

            labels = train_label[train_index]
            test_labels = train_label[test_index]

            train_ds = Pool(data=train, label=labels, cat_features=cat_cols, feature_names=feature_cols)
            test_ds = Pool(data=test, label=test_labels, cat_features=cat_cols, feature_names=feature_cols)

            scale_pos_weight=(labels==0).sum()/(labels==1).sum()
            clf = CatBoostClassifier(**params, verbose=0, scale_pos_weight=scale_pos_weight)
            clf.fit(train_ds)
            res.append(np.mean(roc_auc_score(test_labels, clf.predict_proba(test_ds)[:,1])))
            # res.append(np.mean(clf.predict(test)==np.ravel(test_labels)))
        return np.mean(res)

    # this function runs grid search on several parameters
    def catboost_param_tune(params,train_set,train_label,cat_dims=None,n_splits=3):
        ps = paramsearch(params)
        # search 'border_count', 'l2_leaf_reg' etc. individually 
        #   but 'iterations','learning_rate' together
        max_metric = 0
        best_param = {}
        for prms in chain(ps.grid_search(['border_count']),
                          ps.grid_search(['l2_leaf_reg']),
                          ps.grid_search(['iterations','learning_rate']),
                          ps.grid_search(['depth'])):
            res = crossvaltest(prms,train_set,train_label,cat_dims,n_splits)
            # save the crossvalidation result so that future iterations can reuse the best parameters
            ps.register_result(res,prms)
            if max_metric<res: 
              max_metric = res
              best_param = prms
            print(res,prms,'best:',ps.bestscore(),ps.bestparam())
        # return ps.bestparam()
        return best_param

    return catboost_param_tune(params,train_set,train_label,cat_dims)


# In[142]:


train_ds = Pool(data=X_train, label=y_train, cat_features=cat_cols, feature_names=feature_cols)
test_ds = Pool(data=X_val, label=y_val, cat_features=cat_cols, feature_names=feature_cols)
full_ds = Pool(data=X, label=y, cat_features=cat_cols, feature_names=feature_cols)
train_set = X_train
test_set = X_val


# In[ ]:


bestparams = run_grid_search(train_set, test_set, y_train, y_val)


# In[143]:


bestparams = {'depth': 4, 'iterations': 1000, 'learning_rate': 0.03, 'l2_leaf_reg': 3, 'border_count': 14, 'thread_count': 4}
clf = CatBoostClassifier(**bestparams, verbose=0)
clf.fit(train_set, np.ravel(y_train), cat_features=cat_cols)
score = roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
print(f"{score:.2f}")


# можно было еще попробовать различные способы выбора фич, на основе корреляции, пример

# In[ ]:


# !pip install git+https://github.com/sv1990/CorrelationThreshold/
# from correlation_threshold import CorrelationThreshold

# X = pd.DataFrame(...)

# ct = CorrelationThreshold(r_threshold=0.5, p_threshold=0.05)

# X2 = ct.fit_transform(X)


# или используя стандартный sklearn, rfe и т.п.

# Выпишите какое лучшее качество и с какими параметрами вам удалось получить

# # Предсказания

# In[121]:


best_model = clf


# In[122]:


X_test = pd.read_csv('test.csv')


# In[123]:


def convert_data(data):
  data["TotalSpent"] = data["TotalSpent"].str.strip()
  data.loc[data["TotalSpent"] == "", "TotalSpent"] = "0.0" 
  data["TotalSpent"] = pd.to_numeric(data["TotalSpent"], errors="raise")
  return data


# In[124]:


X_test = convert_data(X_test)


# In[125]:


# X_test["MonthlySpendingGroup"] = (X_test["MonthlySpending"]<40).astype("int")
X_test["ClientPeriodGroup"] = X_test["ClientPeriod"].apply(map_client_period)


# In[126]:


test_ds = Pool(data=X_test[feature_cols].values, cat_features=cat_cols, feature_names=feature_cols)


# In[135]:


submission = pd.read_csv('submission.csv')

result = []
for i in np.random.randint(1,1000,size=10):
  bestparams = {'depth': 4, 'iterations': 500, 'learning_rate': 0.03, 'l2_leaf_reg': 3, 'border_count': 14, 'thread_count': 4}
  clf = CatBoostClassifier(**bestparams, verbose=0)
  clf.fit(full_ds)
  result.append(clf.predict_proba(test_ds)[:,1])


# In[138]:


submission['Churn'] = np.array(result).mean(axis=0)
submission.to_csv('./my_submission.csv', index=False)


# In[ ]:


# submission = pd.read_csv('submission.csv')

# submission['Churn'] = my_pipeline.predict_proba(X_test[feature_cols])[:,1] # best_model.predict_proba(X_test) / best_model.predict(X_test)
# submission.to_csv('./my_submission.csv', index=False)


# In[ ]:


get_ipython().system('kaggle competitions submit advanced-dls-spring-2021 -f my_submission.csv -m ""')


# In[ ]:


##colab
try:
  from google.colab import files
  files.download('my_submission.csv')
except:
  pass


# результат 0.85169 ,что достаточно (поскольку больше 84;))

# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA7gAAACDCAIAAAAoDiD/AAAgAElEQVR4Ae2de3AT19n/+Sf2ZCYpIZmQS9/JNGV4m3ZoO23fhORtmredXmaS4VLCJYbApJ3cTGrakPYlMJm+CS2FEYE4F8CAB9Pww1waYsAQaFAUYSNMjOVLjBFoBDgI29jCF1myvZJWjn5z9tk9OtpdybIxxpcvo2HWezl7zufcvuc5zzk7Lqb8y7RY8QMBEAABEAABEAABEAABEOAExolCuSEYwg8EQAAEQAAEQAAEQAAExjgB0soQyhgbgAAIgAAIgAAIgAAIgEACAQjlBBxjfNiE5IMACIAACIAACIAACHACEMoQyiAAAiAAAiAAAiAAAiBgQgBC2QQKH0bgAARAAARAAARAAARAYMwSgFCGUAYBEAABEACBEU+gKSi1Brtbg93+QJc/0DXZ+b/P2DfleP7fqvpDn3pcnsi1MSt0kHAQuB4CEMojvnG8nuzHsyAAAiAAAiOaAOljEsfi/5Od/6v7rao/1NEbG9GJReRBYOgJQChDKIMACIAACIDAiCTA7cf+QFdrsLspKPGfx9/9qce1qv7QqvpDvzm1kUTzz75cvb26pllqHXq1gTeCwAglAKE8IhvHEVraEG0QAAEQAIHBIsBVMklkY7Ct0TD9GoKhTz0uUS57/N3G+3EGBEDASABCGUIZBEAABEAABEYSgaagRF4WokRuCkoBqS0kXQxJF7+Wz30tn6PjgNTWLLWSYv7U4/rZl6snO//3Z1+u9viZBdooC3AGBEBAJAChPJIaRzHncAwCIAACIDA2CXCVTMlvCkpcHJNENv4fki42BEOt0bAnco1My9DKY7PwINX9JQChDKEMAiAAAiAAAiOGAHlctAZV34lmqfXr8AWdMm5vr61vqapvqdKdJ+9kT+QatyvzcPqrHnA/CIwRAhDKI6ZxHCMlEskEARAAARBIRkCnkgNSmyiFZalOlurqW6rKXLXbq2tOV59sCLjFG76WzwWktoZgyOPvJq28vbqmNRpO9jqcBwEQgFCGUAYBEAABEACBEUCAuyaTb7FOJX8tn2tvrz3rqmj2lra315JR+XT1yfqWKlmqE+Vys9TaFJQ+9bjIWRlKCARAIAUBCOUR0DimyD9cAgEQAAEQGCMERHNyU1AStS9XySSRGwLuhoC7vqWqIeA+66robHbqtDJJbXJWxoZxY6T8IJkDIwChDKEMAiAAAiAAAiOAAK3hI43LV++1t9eSCCZzcki6yLZPrq6hH3kq17dUtbfXkpimg5B0sTUa5kZleF8MTELhqbFA4CYIZW+g6fDW5c/Ne25pcZMOcbJL9S1Vu1f9cc6sWc9kLc3Js9UFb3KL1nL4z9++7c6H30pw7epyvvPTCbc9uPjTho5/TR9/+8Nv1ehSl+LPr4oWjb/jkbfKuhsC7vdnPHDHU9sd/khaG/d0/OvX3/3mw2/VpHNzvWNfzpxpP/zWxAm33/Xgr5bl2a/e6O2B7G/96L7vZ2/xsj3wzZPff1bm4dzsIoFYgQAIgMANJcD9LhqCIW5OlqW6Mhdbt8cdLUgl/+nL8/T71OOqb6k666rgzhhnXRUkrGkTjGH+gWuuCmbMfXn5xqPJesZmb+mG1xbNnJ017YWVbxY5RZHgPvi3J6fN4L+nsjc7/BGeUzx8nSBp7Gg7WbRpyQvzZs7OWris4OPL8V6MHlnywrwZc19esrLQWt/JQxutB5fPHciZM40znDH35VTd+qjrjodaKFNpnr941ZIX5i0tbhL1U9JLAXf+K/MWLivY5/KVVR62ZD+9YJ315m6WfqXj+Iv/efsdT213Rnp4xXBtnjZu/E+eL+kagFB2bZ6mCuVg/QfZT1JNFuHwt+gOqPj+dnWJ7rzxz4uONT+dcNu48T+Z9sLKjbmLfv3db44b/5PZRfXpvMUYWppnHLkLH5/1148vJ1HJwdAAWKX5atwGAiAAAqOJAPldUIsteifzDS7a22tPV5/cXl1DEnmVcrCquoaUtCzVnXVViP7KtAOGusWytofGcCN2Iv/VaS+s3OfyOavsG15btGCd1SSGAbcl++lFW4+Xe/2Oz/b8PWu+qC5O5L/KNMOVRs+VxnKv33OlkYeQVHUEQ+6Df5v2wsrt1TUut3v3qj+KnTJFaccxtmJSd4mHPJoOmoLSRceap7I3W+s7jQxHU0qTpWWohfJFx5qcPJsnci3/FSaUxWglu3TRsSYry7K3g42h2f0d/1qy4NVU2mtIRjO7f//guHtnrj2vbdgecK/5n/HjJr9sC0UGIP64RZnLVn4gIjIep3lbfUvVm4+y6G3xSh5Z9vi7SeuPm/xyHOwN4NYUlBo72ArrpD9YlFPAwSUQAAEQ0AiIQpn7XZCbMjldtLfXlrlqSR//6cvzOccq//TleSaUKw+TyZnsytyzmXZWJkP18NwkrtlbumTBq++e7qHvcl8+d2DJgleNtswu5ztZWRZ7MEy3uQ/+7TfLdnJr2uGVz7648yvqg3Q9ZjLVUd9SteG1RWyOV4Ff31L1f3/Ifvc0s4td6Tj+96z5/FKDYsh745OWpH2cln0j+gb3wb/NWX6I7PQ6hiM6XWlGfqiFshotpWzphHKyS2IOKffUG0V2mqkdrNuaghJ5X8zY5qUwg5e2/XTCbd9feZpVTkH8eQNNjtyFv/7uNyfcftf9339q0dbjvPaeyH+Vzj/4q2UbczXXi2D9mv8Zf8dT2+uCoa+KFt0/4XtLVhZm/ew/Jtx+1+Oz/mqtqHl/xgMU1Is7v2LlVXiX/a0f3flA1sbcRT/81kTmASLUTx5b7mvBxoi7cv7riRlKhY+/lBqCF//zdgrBGCYbtyjxueu+J367uqROaTjIvu6R5YZgyBto4mMG+1s/Gn/PvIJONs9V31KV/8o80fGDxVCJ/4/XHPsg+8lv33bnXfc9MX3LCTHmOAYBEAABEGgIhkQHZd3GyWRLrm+pslYw12Tud0FC2VpRc9ZVoTMn03f7yIvDH+gankL5omPNvBfyyd2C9XdJVGnw0rasLAuzUim9nqgZvIGmwhVzzcUG7yINguTyuQNZWRbu5kH6mwKnS/ZgfEO9wyufTWdSd0SX4RP5r97ENJ4s2rS9usbx2Z5N295fu71on8vHxbqzyn5o995N297fWGgjHxhvoOnozk27K1R/mHrHvjUFVs0Vp/7Q7r37XL7+5sVNEsrBFGJXf4mqCg0W+XiODxD7m+DBup97X1AGcL8L1twI4tW1edr9E773VPbmTUdKPsh+csLtdzHfjGCIHCEe/NWytduLPsh+8offmjj+jkdW1HU1JArlb9925/3ff2rlkcMbXltEOvLJddZN297P+tl/3PlA1t6OBNcF+1s/Gjcu477vZ6/dXpRnv8pLUkMwJLh2JBh3tXtSCWUxzIuONT/81sTHZ/1105GSjbmL1LFB5Bqzryvm6qagFLy0bfr427+/8nRDMMSFsjfQxO4Z/5PfLNvJ47/Fqwp9FZGSrnHjf6JwSIjnYOUawgEBEACBEUqAhDJFnluF6YD2Tv7s6PHtyho+USj/6cvz26trnFV2Z5WdlvGJz1JoYsjDCo4oeSlionk4HtWAe8Nri7ILHXXBULO31JL9dFwZK14Z8xevWrH42RlzX375XWu51x9/UNXKCaqDPA3mLD/k+GwPPSU6IpNVe8PZKPWeJMTnLD9EdiJDyKOkIzu88lnOcOGygiF2yz5ZtGlNgZVe6qyybyv4mLzML587QLrZc6WRztM9js/2kAQi0bx2exG5mDd7SwXR3I+sGQFCmcr9oq3H64Ihn5u5BM2cnXXThXJDMLT79w+Ov2feFq/kDTS9P+OBcZNfVkeZglBuCNYXl1R6/N3NUitZnWnhHXlu0BRSs9TKgqLFfAahrBqtA+43H2WWZmekp1lqJeG79nyCKGeqVFXb+hIQv19pFxo72rjDllKxUwllHqYaz0dW85UQTJorzhtksaYxAL2LxC4XyjQwoOWPTUGJ2ho2k6WwenDxp57INY+/m2777x0NmoLXJ2S0NkNIFwiAAAikJiDKWVHs0nF7e21xSSV3UCa/C1LM26tVizJ9jkR8lt4ohpw6DkN81WjIPLzy2UVbjxujQUvVabUZdxJQbqs/vHX5mgLrpx6Xs8puyX468Sp1MQlCuUFxUJ45O2v+4lX00RZyRCZrHVvJt/JZEovNUivtTMCEsl/10zDGbaSfaQpKJ4s2rd1etONY1enqk7tX/XHaCyuH0v31ZNEmEr5NQSa3yGDc2NFGB1wtOD7bk7vf2RSUmr2lGwvZrg/1LVUffmQ7tHvv7opOj7+73rEvd79zANkxAoRyQzB0+dwBS/bTT06bMWPuy9mFjr5nUviUyg07SPC+UNQe+V2wPBOEMq2c/b8/ZD85bQZ5HbDdMDTVq00HiBbfuGZtOfzn+yd8j9Qnf4RqY1yMCu/iqtRYDuh+7lbFZPq4DPopQjz+UqPrBfedoDjwB9UDxVGb7OtM7/q735/xwPhHVlObwqNEbiTPl3RRmY7/L8SfrNE/nXBbmpt4GJOJMyAAAiAwWgmQnKXGU3S9kKW6zmZnZ7OTVvLRrnA5xyr5DnGk9shNWdxNeZj7KDcFpTQtypfPHVix+NnsQofL7S6rZBOwv11dYq5clTVO5G0slBMTofxM1lJuNm5QzNLcEZm24SJRvmRl4eGty2+iW4KQihtlV9KvNTJ4qtzoOHCh3BAMkT5mMj3g3lbwMbdtM31cY8vdr2x4EnAf2r3XWt9JZ5q9pbn7nR5/98miuEtGv+I8MoQyJclzpZEpS2WS5a0ybRXdDZPCfXLk6pD8K7gKFIWy/a0f3T/he4/P+uva7UWbtr1PKpCrXlEo33XfEzp34QRxqWlrjyw3BSXX5mnq/YLQ5KrUGHPRR7khGHJW2Q8Uf75vw1++fdud6QtlWhE4/pHVn3pc1ooa/qNU7P79g/d9P7us8vCvv/vNX2gLk3mUEtIiZpkQfxZt7U9s6mnMRJwBARAYywSSLeYjodzsLXVW2TcdKSF9TN7Jb5dUv11SvelISVnl4eYaG+lprpWHuVAmH0XyUdbyXa9o6bwjd6EoVWnG0tzkqYi8t8q6E2W0Plidw2djR1vhirnquiCt/2rsaKO+z5G7cDhMcWuIbpRcFsNPZtcX7xnEY1HgNgUlzQWZORxzodwQDMWFcjBEjzg+28OclRVJ7bnSyH02+hu3ESCUWaFfWch95y+fOzB/8Sr+Z38TPLj37/79g3c+kJX1s/9Q97ugKqSpPS6I2UYTkWvc9YJ8GMbdO5M56SojpGSuF2RRZn7PXCgr8zv9tSjT43c+kEVvZGMvqZWpWMVVg5bfjdd8KiiefDEftyir0dY2ymgKSo7P9uw4VkWNBdm/l7ww7677nlhRp1qOuVDmrhfEX+d6wfecplf/946Gwc0mhAYCIAACI52AKJTF7eFIKNc79tU79pH3BWnl7cp+F2WVh51V9uYaW71jX3ONzedmmyiTVg5IbEsiCnZ4LuYTd72guWXTXS8Or3xWXAV+peM43yuj2Vv6Zs4KLpp5gImFQS+U9VJbsChf6Tj+Zs6K+FoxZUOMDWejiQEOhVodwjfW7171R26GJ7fsoRwbkEWZ0tsstZII5q4XnAO5XtCf9Y59m46UHNq91x4M08T+geLPya7M70//YAQI5YaA+4PsJxess5a5amkf5bifvja2Sz/Bg3gn974YNy4j7nchmEVV3+V7Z648cvjozk20WQSJQjLxPvirZZu2vW/JfjrZYr4UQrlfFmVyuvrphNvuuu8Jth97zgqKzIOLP61T9q63v/WjCbffxRYdKivqvn3bnUahrFuDSIv57nhqOy1iaG+v/cMTD4wblzH+kdV8e2kulBuC9SkW83Gh3NDxL7heDGIRRVAgAAKjhoDpB0e+ls/JUh3Xwa7T5fUtzJH0dPVJ+oS1z82+X93Z7KR7SCiTmzJ5cYj6e7ixagpKyfZRvnzuAFfAFx1rnsla+maRs9zrd1bZC1fMZUvxlK+K8MV25KO84bVFc5Yf4j2Ull69UG4Ihk7kv/pU9mbav5n7KNMmIY7chXTpdPXJVG4eN1WfaEkbHMl+Iv9V8th2ud37NvwlK8vCxx6D+yLT0EgZ055dcdcLxYS8dnsR7e4sLuajXbY2bXtf9cQIhphu3vZ+nv2qafh9nhwJQllZxyp+dCdx0mRwykGfpExvUHcjVr4zQo0Ou41blBXv6j88oW7o9ox9k/jFPnF7uH0b/mK6mM9EKCtbsPXboqzUWPHLfP/1xAxaH0npqm+pejtr+oTb77rrvicW5X0ibg/HLcp058kiVfHfP+F7v1m2UzTt07Ybj3x4lqMQhDJzq6f9PSbcftd/PTFDHZ4KrBqCIW507+iNmQLHSRAAARAYswREN2XaSrm9vZaLYDIYc1lMB53NTlmq87kryOQsXaojlUx+F3zXueGJVOlK2Gq85+Y9N3N2Fn2Zj6J60bFm/uJV3LLr+GzP//0he+bsLNrawlrfybuhhgD7Yshz856jD+nxTd+EJJsIZfHze7ov8zUEmFikAJdvPGq2jcbNlCVCugYtGqlo3PjxAFmUySFTcL1gqXOdLt9W8LG4PRwlv1lqPbqTLQFU//SW8r0vBsDnZgnlgeSfuJfhAJJ6gx6hWMXrpFmh8fiZR5TpPeRzbLzUFJRag92tQfOnKC39BeKJXGuWWskArBtskLc+uWQYI8PR8Uut0bDpbjipo5T6Kn8LDkAABEAABIwERDcJ+oq1qJK5Xbmz2elzVwSajkqXVGMz+V1Il9iyP/K7oMZcDND4uuFzhnc9PEpib2I8Nt7PHxysA/GlgxUmwumTgA47/7PPHOd39vkK4w0jSSgbY48z6RPosxilHxTuBAEQAAEQGHoC3PuC2vOA1EbWYulSHf187gqfu6K5xsZ/PncF6WO+ko+5akitpBtEE/XQJwdvBIFBIXCj5Q2E8kBs24OStQgEBEAABEAABPpFgGzA/gD7cFVDMERaOW5X9pZyjwumlb2l7Kcc8GV8HZ3ql8lIJQ/PZXz9YoKbQeCGEoBQhlAGARAAARAAgRFDQOcv0Sy10pI+LpHFg2ZNOtNmF7TTxTDf7OKGih4EDgL9JQChPGIax/5mLe4HARAAARAYlQR0xuCmoERr+2ihHt8AjgQ0X73HZ6h1ZulRiQiJAoHBIgChDKEMAiAAAiAAAiOJAHdWFhd8NwWljk5fSLrIRTMdB6Q2LpH5g/6Autv9YIkJhAMCo5UAhPJIahxHaylEukAABEAABPpLgBuGRbmcLBDaSUlnik52M86DAAhwAhDKEMogAAIgAAIgMCIJcK3sD3SRXBa3waJj0YpMt3EFgAMQAIE+CUAoj8jGsc98xQ0gAAIgAAJjgYBoKiaDcbL/0zE8jwViSCMI9IsAhDKEMgiAAAiAAAiMeAL8M1XczExfreKW5n6JA9wMAiBABCCUR3zjiKIMAiAAAiAAAiAAAiBwIwhAKEMogwAIgAAIgAAIgAAIgIAJAQhlEyg3YkSCMEEABEAABEAABEAABEYWgbhQpiP8DwIgAAIgAAIgAAIgAAIgwAmMi8Vi/A8cgAAIgAAIgAAIgAAIgAAIEAEIZSuKAgiAAAiAAAiAAAiAAAgYCUAoQyiDAAiAAAiAAAiAAAiAgAkBCGUTKMbxBM6AAAiAAAiAAAiAAAiMNQIQyhDKIAACIAACIAACIAACIGBCAELZBMpYGy0hvSAAAiAAAiAAAiAAAkYCEMoQyiAAAiAAAiAAAiAAAiBgQgBC2QSKcTyBMyAAAiAAAiAAAiAAAmONAIQyhDIIgAAIgAAIgAAIgAAImBCAUDaBMtZGS0gvCIAACIAACIAACICAkQCEMoQyCIAACIAACIAACIAACJgQgFA2gWIcT+AMCIAACIAACIAACIDAWCMAoQyhDAIgAAIgAAIgAAIgAAImBCCUTaCMtdES0gsCIAACIAACIAACIGAkAKEMoQwCIAACIAACIAACIAACJgQglE2gGMcTOAMCIAACIAACIAACIDDWCEAoQyiDAAiAAAiAAAiAAAiAgAkBCGUTKGNttIT0ggAIgAAIgAAIgAAIGAlAKEMogwAIgAAIgAAIgAAIgIAJgaEQylPsfkkOf3b0eKbFRlL91nVfvN4Y7Q23LPjAJE5GOZ/eGdtcVygm/PNcacwudEywDOIrrBnrKwo6I8JL2KHrdPngviV1em9d98XsovrsQkfq23AVBEAABEAABEAABEDgeggMhVDOWF/xXpvcG275xTpVs046cq01Gj5hP3U9UTc8y4RyrxzccazqFduFV2wXHP4I0+LaSw33pxLQD5cFesMtL7+rv4eEck9n2yu2CznHKulFi7YeTxF4hqU053I01OiZPBiS/dZ1X+RcjspRuVcObiy08bFHigjgEgiAAAiAAAiAAAiAwAAIDIVQzrTY7t/fyJXxreu+eK9N7mr3PjkgCZs8kYpQDrcowTIFOfGQr6M39uFHqhk7+YN6NZxpsf6yqsdUZJNQli7V3cNUry0dQ/IgCuVb133x0rlubs+GVu5XnuJmEAABEAABEAABEOgXgSESymQH7Q23zH3XOunItY7emOKJYc2wlP7c1uzwRyQ53NHp23GsShGg1jv2XGmNhrnGnfxvf68cXLbNesv6M0WRqMvt3uKVeuXg8o1HBZOqKJSZ9uWBTLDoX5RffIo07q3rvni+qr0uyHw2Ojp9ufudmRbr9DMSF6M+d4WohrlQFk8qxG1khF5a3OSM9MhR2XOl8cl1zFXjvTaZh3ag+PNb1p+xhSJlrlpKwrJt1qJI1OeupYSTqv46fEGxvidIfJ1KpjChlftV3HEzCIAACIAACIAACKRPYGiEsipbPbLscrsLOiOhRs+P1xzLtFin2P0dvTGX2z27qH7t+W7uysw1LqWEhPLyjUdJZUpy2FpRk3Os8gdrmVlXS60qlH+7umSCxXp3wVl6ERmYp9j9rVH21Oyi+hV1XZoKt/HzM3eVr24I94ZbcvJs926teOlcd68czN3vfCyvVNTEXCg/tPnkAxbrQ5tPTnzHTtblX1YxfdzY0ZZzrPL5EvYKkr+PfHiWLOjZhY6puarWl6OytaLmFduFH685Nv2MFLde57ltoYjLWSm+NNNiNVXJ0Mpa1pvMCeASCIAACIAACIAACFwngaETyhmWUlps1ysHFVOxjayt5LxLRt/VDeGudu/U3LgxmJKnsyhzz4fExDOhzG23sViMW1vpRdw2nGEpfb0xSoGQSFWcjG0ZltJHC1RLM7leJPNRNn0LsyirRm4m3393Mfp1+MLUXGbMFnyUbaT1NRMycxERRwWTjlyT5LDifBwXf7eu++J3F6P8pR296mFrNExHPKWJQOIh4DwIgAAIgAAIgAAIgEB/CQydUGYxy3M7Iz3xZW3Kn9WlJzWrsI17Bk885NOMvkztiRZl5npxutwsnapFObvQMXNX+eMHL6w9z6zCTHQqZlouNOkg1OghRe6R2cK4cq//7ZJqxUTN3kh+FMaFgGRRDjV6ZhfVz9xVTr+HNp+coD2iaWsWGfLDThTKmveIs1JLAhswFHRGfO5aktcUMe0qi8wde66QOO6Vg8Ulla83MtEsR+Udx6q4X4f2FLevQyWDAAiAAAiAAAiAAAhcF4EhFcrkYcwtu6RfBdWrCuWX39VblKfY4z7K5Jkg6kjtWBXK2mI+VZL63LV3F5x1RnpczspHC06JP3ILvndrxSu2C1u8Ums03NPZtijvE4PqjSPmrhcTLFbFO4IL07jKV+KjRuYX63QWZRYrQxJsZNievuWE6TCALOJyVD5Q/Pnt/zjFhfKHH6lWebqkjDd4fOJx1vjgDAiAAAiAAAiAAAiAQD8I3EyhzK2ztG/areu+WN0Q5obe1mj4QPHnJP6YjlS8GtTFfCktynwzDS5qb1l/hvyVH1C2qsiwlD5+8AK5W0zJPzVzl7oL8v37G/kqQ9W2bdjmmYep8yHOtKiL+bhFmbSvKJTJXG2ahDv2XPHI8u6KzmapdU2BSf5lWEof2nwyU1mVSEI5FmMbepDLCl2CIAYBEAABEAABEAABEBhEAjdTKGda2Fo6SQ7zxXxyVKbvkpCa7A23vF1SvaKuqy7INkimXS8M5lguKxUjrrCP8havRKsDyULMX0SL+U7YT5HbNG29TKsJNWdf5uwhyeEyV63uux4klHX7KNM93G9EyZ6460WmRY3Y9uqa6VtOmFmU45tjhBo93P3DNJvJwZpcL5SBBE8+DkAABEAABEAABEAABAaTwM0VyvFd2+SoLG4Pl2mxTvqo3uGPyFG2UcbzJV1cKJs6J3BtSgqS/vcHuopLKhUrcsKLAlJb/Pz6CtLNtGHFpiNsx4xMC3NpeL0x2hAM+dy1ovGYhLLO3dnnrjBalMlHmb6xcnfB2b0dIUlmNnJTi3KmhXluxGIxxWM7VQaLQplvn2cqqXESBEAABEAABEAABEDgeggMqVDuT0Tpm3P0RY80/W7VR7SlgankZmJM6EHaaa7Pd/Gb0w9ff6eovHlMaNOMZdv0N/MbtAPm0Exf5svJ6zO2fYaGG0AABEAABEAABEAABMwJDFuhbB5dTSyOrqt57ldsF2yhiHSpjuzffSVzEMR6X68YXYQH4+PhIAYCIAACIAACIDDWCEAo33xFOMXupy/5LTCsHRxrxRHpBQEQAAEQAAEQAIHhQwBC+eYL5eFTGhATEAABEAABEAABEAABTgBCGUIZBEAABEAABEAABEAABEwIQCibQOHDCByAAAiAAAiAAAiAAAiMWQIQyhDKIAACIAACIAACIAACIGBCAELZBMqYHTYh4SAAAiAAAiAAAiAAApwAhDKEMgiAAAiAAAiAAAiAAAiYEIBQNoHChxE4AAEQAAEQAAEQAAEQGLMEIJQhlEEABEAABEAABEAABEDAhACEsgmUMTtsQsJBAARAAARAAARAAAQ4AQhlCGUQAAEQAK/aAQAAACAASURBVAEQAAEQAAEQMCEAoWwChQ8jcAACIAACIAACIAACIDBmCUAoQyiDAAiAAAiAAAgMUwIzd5Xvc/k8/u6GYGhQfrEx/K81Gi73+mfuKs+02Mas8O1vwiGUh2nT0N+MxP0gAAIgAAIgMMoIzNxVPijiWAxkDOvkeNKhldOvKRDKEMogAAIgAAIgAALDkIBtn8snatxBOY6rxTF8VO71p68Ux/idEMrDsGlAlEAABEAABEAABKyD6HHBRfYYlsfxpLdGw/C+SHMAAKGMlggEQAAEQAAEQGA4EuDqdhAP4mpxbB+lKRNxG4TycGwaUC5BAARAAARAAAQGUR/zoMa2PI6nHqUrTQIQyhDKIAACIAACIAACw5EAV7eDeBCXimP7KE2ZiNsglIdj04ByCQIgAAIgAAIgMIj6mAc1tuVxPPUoXWkSgFCGUAYBEAABEAABEBiOBLi6HcSDuFQc20dpykTcBqE8HJsGlEsQAAEQAAEQAIH+6mOPv9ta3+nwR1I8OLblcTz1KF1pEoBQhlAGARAAARAAARAYjgSMerfc669L8om+Nz5p+c4/A5OU3wy77+PL0runez6+LOkCiUvFsX2UpkzEbRDKw7FpQLkEARAAARAAARDQadyGYOjFnV/NsPuMWnl3RSdXyaSVJ/0zQHJZF4ipPA40Hf171vwnp82YM2sW/dTj5Yc6ek2fGKKTLYf/nJVlcUZ6dO/7Onrakv300uIm3Xn+Z68cPLzy2d8s2+mRZX5SPEDpSpMAhDKEMgiAAAiAAAiAwHAkoNO4TUHpxZ1fTfpnYOqRq6KpuLGjbcY2r1Eof+efgb0doaZgglFZFIv8uDfc4qyyn7Cfcny25+2s6QuXFVgrak5Xnyz3+uWoudDkzw7gIHhp25IFr244GzU+q7uUVCiHL+xe9cd3T/cki15vuOXwymcXLiuAUE5TECe7bSiE8uR/+wNSW06ejUciw1L6emPU566dbLlZNdP2y6oe6VLdhAFF4NZ1Xyjxr7inP48P7CkObZgc3LHnSrPUumCdScbdXXB2b0dIksP+QFd+8Skj2wxL6c9tzXVBdk9jR9uybVb+ZaCHywJie9EbbnlSe8Wkj+r3doTEM5kW29QjVx3+iByVtXDU+NxdcHaLV+rojfkDXcUllddZwG5Zf2ZvR+jDj+JFV8sFVn4CVy/9Qoukdj6OZYrdH2r0pI7A/fsbm6VWhUP8QWNQ13lmdBS864SQ5uO3rvvipXPdHb0xn7v2AVa7jVl/A3MqzUj267Yxmfu2n9uaPbLc09m24AN9fvXZTGVarJM+qrcHw8amrJ/NlPrq+/c3emTZ5azkTWLG+oqXznV7ZLlXDrrcbqWt66OkqY/4u6nRyy8+1a/ep19lRru571aOuvIT9lPaI3Hat6w/UxSJfnb0uPFSv87ohHJDMJRnvyoajDecjTr8ERLK/Dw/KOg0cVYW+xrj8dfR029nTX9x51exWCyZBjU+1d8zOjUsPq67dHFXjqlFmaKXIoZkUZ6z/FBrNCyGz4/7lRFj+eahEMq3rvtidUPY567lFXvSkWuaPuijdbhheXNdQjnDUjrD7tt0pKRf0RvYU/16xRDcnFQo57mLIlGXs/LRglMz7D6PLBv15RS73x/oyt3vfLTg1PNV7R2dPq0bs00/I/nctY8WnJqSz34PbT45wWIlYd0aDZd7/aJQpvKTX3zqkQ/PvnSuO94d5rltoYjL7Z65q/zxgxcc/ojYOQ0ATsb6iiRCmXWl26trUujgdITy3QVnt1fXKGr7BlaE0VHw+pt9GesrCjojxkKYOpyJh3y94ZY3i5yP5ZWmvnOkXB2LuZ/ndkZ6iktYW8S1qZpfaTRT39he74z0WCtqDE1Z/5opeiN1f7FYzOWspB4ww1I61xXqavdmFzoe+fDse21y4OqlH6yNS0xj0aIRe09n25tFzpm7yp+vavfI8gm7IXX9Mdzwt3xje70ncu3ld00i0Gcrd1OEcrnXP2Obl0thOvjOPwNGc/IkxZxslNpcKZoeiEI5Fovp/nTkLnwma+nHlyW69Pes+eT8IEt1+zb85bl5zz2TtXT5xqMNwRAF3htuOVm0ackL856b99ySlYUOfyQWi3U53+EOHjoha7xEFuWyysMrFj87Z9asJSsL65TAKWJvfNISi8V65aD74N+WvDBv5uys+YtX7XP56KQjd6EufDHJvAzgIDWBoRDKmRYrDanXFLCqmLG+4r22hOF16ijemKvXJZRvTJRM2qlh+KJkQvnhskBXu3dqLhnhWI8iXaoTdWSGpTTnctR1ulzpLWykYzR7g41dclbq0puxvmKLV1q2zTrxkC8kXSS7y63rvnivTVYeZOJS+NM6xe7vavf+Yp1V6R1tZK+da9YB6F6U7M/kFuW+MysdoawYLG2jwGyZDOBNPD8woTz532weILVquYmJwqvTIZBC+fXZTGVarA+XBYQyYJvrCvnctdSkpN9M8XhOOnKto9O3xStxoUymVmUIx1owim3KaSXWnHa1ezXDM3tqip3N06Z8qu82iiKZAhdPRbKDmyKUX9z5VTJZrFPPk/4ZsIWuy6JMspLLza/DFwpXzJ0zaxbJ0y7nO0sWvLrFK30dvpD/yjzy1qh37LNkP71gnbWjl+lXR+7C+YtXbTpS4qyy7171x6wsi8Mf6Q23uA/+7bl5z+Xud3r83aJ4/Tp8QXep5fCfZ87OWrKy8EDx5yeLNuXMmbbgA6sclb+Onv571vw3PmmRo3LL4T8/k7V005GSs64KR+7CrCyLLcQUOcU8mY91smzFeR2BIRLKGZbS312MknKaYveLpsSf25od/gjNxefud5IBIFFkUFNVMUFrU974pMUjywbHifh0fE9nW34xTQbZfneRiTMKljQ6aSxyvZjrCjUongBlrtofrzmWabGRNtpxrMoWYrHyXGl8LK90dUO4ozdmGiwZbMgHyBto0jxM+o7MpI/qP74sSXK4p7OtuKRSmee13rHniidybWlxU10wJEdlz5VGU51Hs4dyVJbksLWihp7VOSTk7ndSZt+67guyQFByFm1V58II8kvnulujYZo741HyB7qsFXFzKZ+I7Oj0LS1u8gaadLHiIpgXL1VPC/OeunuEvGBid3WDGgceAh1QxpGdT3XGUMxFZr4fYpfGOolb1p85HgluLEymRBPun3jI1xoNkwGSR5UsyjuOVe3tYNnhD3Tx0BKKaJ577XmGUZLDZa5aZbTAerJQo+f5ki6PLMtR2eV2G+WX2EUlydN4b5exvkLMx+xCB3Xev7sYLXPVrj3PHAZ6Ott2HKsiaALJeC2gN6o1SIntIx8yhxmaz12U90mmxUb3LC1uIucWz5VGrcwo4bjdW7ys3CrjXjbHrau/NHQRZmNVzvcooxoxCUplSbPGsQGY6G/DfXtUzoqNTY7KZa5axlkpJNyio5Ycwd7Gi7pQ+1g8eY/lOl0uALSS3OHFnoyONPI3C0qtyNxKN/GQr6vd++M1x6h5ybMzxyFN+qhZzFse8l/q6PTxKpyQa4rDGH8pr6opsItt4N0FZ8WySrMZ9GpufafUUVOWTrFcUdfVGg1TIY87I5lVCvJqSNbuiQ37LevPGOMp5ggNksXiRKV00pFrfDI61OjR2kYGmddrHo6xmeJCWRvkx1uJ/jVTSmHj5gCWBZpFWRDKLFZEe/nGozxWugPKHW4aoKsZ6ytWN1B7ZVIrTWuK6PnGveMmHvJxFWXoUtVGjFCYlhwSytaKGrH9IcM5RftA8ee65PT3T6M9mBbtmdqPdUJ5it1vfJzbenll1x3oTMixWIxsuvZgONB0lDkWv7bot6tL5Kh8cVfOvBfyPbLc5XyHFDCVPfKd+PiyRAsEuSOyaAPW+VeIcdBdurgr55mspfYgq2JyVHbkLqSXiqGRewaJ41452BpUxTeEcn/Lm+n9QySU2bvz3McjwR3Hqgo6I7wTJffl/OJTfJJLESIJ9TPTQhNemlCWmXycvuWE2ALy5qa4pPKhzSenHrnqiVyjbljsJARxxizK5B82fcsJ0grkHJKxvqIoEm3saFu09Th1Er1yML/41MR37OKwngdLSXizyDnxHftcVyhw9dKP1xyjti9FZOiGMhdzNnj84AXmMMA6ZkWjyExUTd9y4t6tzPQuuqxQFlIqzroqHssrJQlLPdykI9cCEhshTMlnzg/k3CJO803JPyU6KkyxszUK1oqax/JKH1Da6+ORIM1XPn7wQlEkStlEUaXbyJ+hN9yi06kCWE3YKUqFlAQveRTD7ELHxHfsP7c1N0utpDyoO/FcaST3ZZfbHe9ulf6GdAYZVOh4dhFzImyNspEMZTTv4bgeZaMOWU7RUlNQpBWmn5FYT6/Io1vWn7GFIhsLmdm7KBLt6PS9WeQkegGpjeLMhTIJFPL3eOTDswWdEcoywlvmqp2+5QRxqy49yVHQAQmgBevYNEtBZ6TMVavLU+F+1mHTdO1Dm0/OdYVCjR6y3//uYpSGcJwqSVjxWV5c2RuVAkYv8shyR6dv0dbj925l/S7FnO4JXL00fcsJKtVd7V4l1axLpuowJZ+5SCarv7+s6uE2OZIFSi6wihy4eim70EEwOzp9c99laU+jxjGp2iy15u5nGUG+PbytoGL8g7XMH8bhj5Ck4O81DBtUaWKsfZkWG2Xr1FyalNAKszLosoXijhwPlwXEmm4Miucs5QIXypRYno9i3MTCdo/FSo0YDcwoR3i7943t9caqOiHRFMqTn2mJD5OonvKy+l4bMzdMVmb5ElyMlMqbk2e7dd0X6RRLKirUXpGzU7JKkbrd4wnk8aTGmUqmplzVTEnesrGpJPIlEPHyyUyleGg5a9ZMEd4dx6ruUZyVnZEesv72p5lSw6dyMjXXKgpl6tEIGvnE+9zK6C6JQzxBS2481tfKZDVl0pFrrcHu/GLm2Db1yFXu/ZgMF9mtaaEFz5GZu8rJXUQtOcpyo15Z7ThoCQoRpvKcovkVGigtO4ShLL+qU7pNQakpKIluyjpxLP750rlu3TI+Ck1UpcZjo1Amvfvu6Z4u5zsLlxVcdKxhUjVy7fDKZ3+7uiQWi13clTNn1ixxx4yZs7Po/riLhbKZxszZWeT9rFPDYjR0l0imkwimd3GhzC3KFMNnspYuWVlYXFLJBz8QyrwgXc/BEApli+LtIIdpcpxPmivqgWx+6nz9A8q8krAQisb0FdzWZdpkkG1AM+HYlCaSBcslQmJDqQjl+IoxG388cRzMbtPMEjYyUylD/3iwunk6app5aEremERG8ExgkaSmbe674jQcm1+j/pUslDybScZR283uecdOMFc3hDnMDEvp4wcvMBOL0hNodlDW873XxpzbqBHkeUFt91kXG43Qi5iovXppaq6VoqoMS1hUHy4LGIVyIjQlBGVcpBPKGZbSl86pI91eOaglgcXqpXPd1ooa8i3e28FUoGLgVyMjul5MPOTj8w+PfHh2RV0Xi49iuuaafrLFencB06xyNJVQpsHbmgLVf2N3RWd7O1tgeseeK2Q1T7TlsAJwPBKkRHGhzEwyiuAjbncXnM05xhYRcj8Q8q+gGQzupk83k5x6+V1m/BZ1GOUpz3E6yLCw8Yz4IB8K0tQw8+pW/JoSpAB7JF5cacqCyvDt/zi1uiHMTad82ECyjEZfmRY2WuC+Lr+7GPW51UJCZYkXOSpC0qU6GnRpI1VWhlU+Cj1tVKMGe6D488TCY17jyJgndLqsraDiKnBmKeUT5SRrtFSo3BR6ajujCK+E2pepSH8u8RP5CyFb2BQTJTx1RdaaI7Uic4uyIYNY9G5Zf6agkwbMLCEkBClnqT1R0LE7edophlRVaXzuiVxTmkdbHLuQ+5TF2sIA1e6+sVA16mus2HDdGenJyWPnUxdL1UYbX6ZGbZ2Q6UqJ5ZUiNS7esE885Etw29WEe0KOJG/ZeLVKuF8hvLcjJJQi1Xyja6YyLWyehEsNl1Od7utXM0UZWhSJUitHQpm3rpTXpI16wy2UZbrY8j+TJUe7gdVusVaubggLaYzXlEyLbeI7dooD1Q66LUX4YivX1e41lhyyKIvWHN4hiq/QoipWw3SPdUKZ/txd0Ul7X4iyWHf8nSR+FwOwKPeGWwpXzM3JszlyF7648yuyKxeXVP49a/5bZaxHI4Outb4zcPWS50pjY0eb50qjHGWW5meylubZr3oDTZ4rjXTVH+iKxWLBS9uem/ccNzb3KZT59nD0Lo/MXC/ezppOrhdyVA41ehyf7dm34S85c6bNeyGfnKGba2zFJZVi4OLx9eTLmHp2KIUya5VsIbL3qB0D2e04cd5V8/qpXGJVnRqCFFWapqIkOfypx/V2SbVmVoxLBKNQli7VceEiGhEF44oi7vltSk9PjRrX39/YXm8LRfyBrn0uX3ahgwJMHZnb/3Hq9cao0tFqjYWmwHQJ5EA4Ig5EktkStx3HqmjVEY9/4p2sx0r0lIj39yJkau9oZkeSmQuBHJV7wy2/XV2iiyr12QOwKGdYSqefkWg9Ci3m88gyGQUpUXGH3XgXqPIhDtyiLMlszpG3+O+1yRpMdXae1jEUl1Rqy7nYqMP44w52dxec9USuzdjmPR4JLlhn/WVVDzX9usV8ZKXjhkwayz1cFhALUqZFNUaKeMnaTSJSjIaQ3Yp6S8xT8U6aOH784IW159l3p+qCoV45qCuKfLyk6DAxyfFaIL6R9kPQ0KlibmquOlrjIk/V2YoY4sWedECy+itMUquj3AnK8KOjN0aliwqYHGVjtkTI5jWOyMdiCY/TTHEiZ2ormEdpMqFMmc5TzSBrtS+lUFbF35PrVIP0sm1sKl9XO3hQxJlXEyrAU3NZrITmJaFMGi8xg72yzEsMLVlVnfuulUY+yjA4jl0cJulGazTU+ezocf2rlQpIoykmbVMWy0lHrnlktvnM9uqaRVuPK7WSDaeNlaJPXLzI0eYSvCGS5HCvTD5UcWLUEHFnGHGMJBTy+P2J7b92XkmpTihP/re/WWrdcazq0YJTs4tY815dytYWK/WRjazUX8pmipuNqSea6wpx1wsaedLUCg312T5CgpearuLrLMoTD/m4bwnZ73W1sijCdhwTK5rmU2Gb9FH9n748v8/FNiGWZFVPJ8MlWpSTlhxDFeA9zo0Tyk1ByePvNq7n0wnl1xujpiJ7AEKZpPD8xatWLH723dNsLvrwymffzFkxf/EqWtXX5XxnxtyX6TgWi4UaPWWu2o7eGBl6yaGZ9UrhFtfpcnJK1pmNRf3KLr0wj2to3a4X3N9D9FEONXrKvX4KJHhpW86caaTgxWCNx7qShj+TERhSoUy9nWa3YBaUokhUkEpqV/2LdXrXC8VKpLleRK7x7keXKjKj/unL8+T/pzQ9cYmQ2FAm9seKsYEiQxNGWiQTb1NaRlN1QgrGE7nGJ9FSR8bYvzojPWsKVI3CE0hmIZ0fgpJq271bK/705Xny9iObkA4mwaHuRPDRTCWUXc5K2nSC/z/BYtVF1Vwoawv1eI6ot4mtf7xfUXsaMi/xsQp/VjBhqv2Z6KNMIfM+VewgiUyGpfShzSeZ8TVuwtT6RYNcfrgs4HPXTrH7SRmvVtz+VjeESWuSsVMrDAklluszXRfCU8FvoDMkHbhJmE6KAijTos9THhSp5JzLzMVie3VNdqHj8YMXnJEe3eRGOkKZLMpUwHTChYs5Q8epVCIzoawrcsK4TvVheGjzSU0XqpM25HfBC9hkzc6nQTavcTQOJLci/uxDm5krSyLnAQplqn2phTJvvvgbdQBZfmnCS8eQyHCLspbYhGKZaFlnlyb/my1O/cHahKELvVRXVWmXGE5DxC4KZZ1+5UJZV84pFZoBO1WxpCJ679aK2UX1W7wSrXaYYGFDTZ1QpjKsa0x0uLR2j5WBUKOH74FDOa6rOylaNh18Xo/S8VGmMd7p6pOKGma5INjmE/IrdTNF6lbr3eI1iAfI22Rxlo9HVTygYYxi/WWj31vXffFoAXNWZGviFYc9nVC2hSKmNWXqkatk0Mk5VjlzV3lBZyR9i3KykmOsAjxfMiylz1e1K8seEriJSUvnOJnY9fi73/ik5b93NJg6K5N3VrJnjZJRPGN0vaB9KnLmTMvKspCvMPla0IYSbF2dfC7/lXnzF68qLqksqzy84bVF5B1Bi/meyVqau995uvrk7lV/5DtmBJqOvp01fcnKQtqhQowAyWt+KYXrBbcokx/zjmNVZ10V+zb8hd5Cgh77KKdTzFLfczOFsjB1SxWJaTiyuk06co3WvvDmtU+LMjUfSmrVmURqWfhULJ8LIw3EfJTDLdwgQTsk5OSpuzFoBkt9t308ojfj3buV+QqTvYEa6GXb1LbMGBneotFuaErTz9o+Wir05Do26S/uUjzxkK8hWK8TyhmW0in5p+hZavp97orb/8HaTW4nM3O9YJDFRpn39xRPXZTu3VoxfcsJkqGiKdTU9YIspuS1yUMTnGfYq6nnEIw3anfIZsDz3PtczFNWeVYVuFofw54lnUFdi246mLorSvikI9dcbjf3VOFT0mqwBpVMsfIGmj6+LPFS4bnSeDwSJJWgs7SJQztOT9eP8llmfgO9PYVQfvldZps05qlmxGIEEmcMmOj0RK4t33iUlsny5aqCKZc9pf3iw0VRQ+g6uQShLPh2k4oiPrwA87LEi5y+qCijFJpzVAuwcoZna4aldOau8ge0MaqmHc1rnLGtmJJ/iuZSEjn3LZTJ1Cduk8xrX2qhzB/k4yh+xliRibOmNZlwpAZNV6K0DGI5RUN0vqEh5azY7nEdmaSqKtmtQH7jk5aOTp/WbsRzX1dWaSRJvvgFnfG5PmqFWGPYV7Gk0Z02fae6G03N1YtLXil0Mefkv7G9Xtfuie5MGesrZu6i3XJ4kVbHJLw4iS0bNTXCWDr+FHcuV8iz0qJrpqg94atoMpWZkICk7MesNFM8Tzk9noliMzXFrpr3RAFEvv6iIxnv4LgHFA9NOGDxTFz6ydzAHH5SuvH8Na2VVFMSKzvrIosi0fSFctKSo1iUhQ6CTSYkcV6K54KQtL5PJhO7DcHQ7opOU7vyL6t6PP7uFA+KmWI8NhXKdHKO9pU+stq+uPMrbt3n28OJ28CRFZm2h5sza9bCZQVcFvMN3eYsP6T7IIh4qSEYIqFs6nrBfZRD0sXDW5fnzJmm2x7u8MpneZyNKe1XRozlm2+mUCaBGJDaaGNdYYGOuqatuIQZOOk8+epR96N1GAl1LNkKhil2f08ne8WU/FMr6ro6emMkoJnNQ1n1T+tFaA/ge9Kzb4lGGuoCswsdtMSKxPf9+xspXTz+SoMeb9GoKaetOh8/eMEeZC6PfFsP3sRTy6t1eFp6FatVcUnlY3nMEZmezbTYaGUVX+3E12rwRWDGxXxiJ0FRovVkjx+8sLcjRD6gPKqP5bHPhdDWNiZZoO1hzNdlkvS5f3+jtb6TOt3XG6M04fhowamf25qdkR7qkEje8bnILV7We+l8lIV9lJmWIheOKfmnyM5K2zWQb0+Zq3bmrnJaVqLEQXRC0BhqIpLmB3tl8jxmArSjl02cUd+vs7SZCmXqWfkCKXExn4g3hVBmMJU8pRWTj3x4lmZ7uU2LVBQzILndj+WV0pLBvlwvxJQmFDyPNieT2HeqoxHV9UKZTCfrb87lqLiYj4vy5PWXvZqGcK3RsKj85rpCPZ1tFOzzVe3qelNtw2OatU8wRipYlDkcJsL8ga4dx6qm5LPCoy3TVE3X2kqvuFAm5UTrYnVNPC/StJSW1z6qRDz3dU+RZmoIhsTPJyULikoOvZ1W69LiS12JEl9Bl2g9LlWi1ijbWoS8VsSdbpNVVfJsfr0xSti1uZp47uvKanwxn7IrUVe7d9HW448fvFDQGemVlbGioakRFSSXZZTMKflsrE4jEN2LeKVIhosGfrzdo8d9blaRp+QzT3pt68l4qU6+mC/BAC8SZsdJmqlb1p/Z4pWouaDmhXolagkpUf1qprjdd0o+s/7SB7YeLVC+EqLFYeaucnJCo4wm32hxqM8jT2WD76M8w+5j30O5VKc0U/H8pftNawrhCly9REhX1HXF128ogyvqUPgbeVDUiOkylJccct+irztRq84rJiHt79cGdBHItFh1etfj786zX32+pMvUlvydfwZW1HXpHjH+aZSMAzjDJXLqZ8mhMfU91391YG8x0sYZUwI3WSjThjV8e6lNR0o0K5r6aSXa72lFXZcolHl7KiaJb9MmR9lCfr57FP/UFu2cpRlOmNjyuWunn2HThaSYSZklmnzM7VuiUM5YX8F3R/JcaaRNu8TI8I14xKf4NklylH0+StseTrUUchlK43i9UFb2ybIH2Y51tDmUZtFR9+oiAnxvKb49XCwWE7b60s1Zs/VDtOcUOZLy/fIoqpRH3kDTizu/4kpL5J+prJ+jjcZoEwbKSjLVk+GHWNHomUSP1pezDmztefapKtpkiluF6RVGG8wMu88ZYZ/uJOY8JvQZP9p0T/h4FcvHwNVLmpyKd7dkB+WLGkkKc+tOYmEwd72ghNNWVlRclfzSCTi2Oatgd1EjINh3GXyHP0LwrRU11AXydHG8sViso9OXZ7+awvVCp2bEgie8Ue9im2BRjlx745MWmmQUyoy+S05ef1kCJ//bL8lhbvMjXUX7ecViscYO9vUEGgNo7hmsEOqEsjaHwy7xTa+oLlPhSWZRpnJrDzKNRVt2iDCpqOtqX58WZVqDpcvHZEHRjjRURJ+vaqf5lhRCmcZstC+QHJVbg928PTQYCJJWVUpCokdvQq6ZbvLFyOS5CzrVbTqXFjfxtZhTj1w1a2riNejugrMfX2bfwqSaSwsJklQK9pQpLkMC2WJc2nGMAJq68CZr2cRCLmY6HfMN78RmiszDfNzOmxfafzPeHPWnmRJezSZL+YiR4NAHRCl1bxap+6L+sqpHNKULfoJZdwAABA1JREFUIbAJhxV1XbRtKAlTrYlIyF/lEfOaQsqVtzDaukY2b0AjT6O3jFi5TEsODbZP2NlIgG9PqbbqynjA0BbFS46YuhTHRpnr8XfvrujkWvk7yuesny/pWnu+uy7I9nvt83f9wnR0hJACOy6JBIZUKIsv7utYWzDBLH90nH4FM72fB8jti8Yz6b9Cd6fpG3X3JPvzep5NFqbx/BC8Jdkrkp03RjLFGWMgxjP8ceMl1lGJe3r0Vfx4UP09ML66vyEku//Ghax/o24e/Iax0r837Rf1C0Xqm1Nf1ceQ7GqKF4r+Uv+bKX0IgobmbZT+nrQRpfNg/9KexqspQGPkTV9kejKdaOvu6W84Ke4XY063iWd07zX9M1ng7LxmA9I92N9X0OPJXmQMPFn4aYUgCmWhAOie1f2pi8P1/tmn6h3ADaND5l5/KoQ8vd5sGt1BDVuhjGwDgUEjQHOm5JA3uuvzoKTOaN4blGBHeCBsTdsMu49vpD3oydFNXwx6+AgQBPpFgO+tnkTiD1r7nDpWA9DBfT5y/RJzdISQmjyucgIQykNU2zlxHAw9AbLVXf/666GP+U15Y+qZ65sSpZv+Uu7LTosNbkR8BIsyGiUQuOkEmEdHR6cvvnhRW9dxIwp/ijD7VL0DuGF0yNzrT0UK7LgkEoBQvuntESIAAiAw/Anc2PllsVHGMQiAACcwAB3c5yPXLzFHRwgcMg5SE4BQHv49NGIIAiAAAiAAAmORQJ+qdwA3jA6Ze/2pSK0OcZUTgFAei00Pz34cgAAIgAAIgMCwJTAAHdznI9cvMUdHCMM204dbxCCUIZRBAARAAARAAASGI4HUnw7pUxOb3jA6ZO51pqI1Gha36h9u2nRYxQdCeTg2DcOqiCAyIAACIAACIHBTCOxz+ZqCkqneHfDJ65SYo+Pxcq//pmToSHwphDKEMgiAAAiAAAiAwHAkMHNX+YAFcbIHR4fSvc5UzNxVDotymqodQnk4Ng1pZh5uAwEQAAEQAIHRS4DtNjNzV/k+l28QfTCuU2KO6Mdbo+Fyr19Tycm+RwNdlEAAQjkBx+htbpBMEAABEAABEBhxBLAz46BnGZD2DymEcv94QUmDAAiAAAiAAAiAAAiMEQIQyhDKIAACIAACIAACIAACIGBCAELZBMoYGSQhmSAAAiAAAiAAAiAAAikIQChDKIMACIAACIAACIAACICACQEIZRMoKQYWuAQCIAACIAACIAACIDBGCEAoQyiDAAiAAAiAAAiAAAiAgAkBCGUTKGNkkIRkggAIgAAIgAAIgAAIpCAAoQyhDAIgAAIgAAIgAAIgAAImBCCUTaCkGFjgEgiAAAiAAAiAAAiAwBghAKEMoQwCIAACIAACIAACIAACJgT+PywXmDpNfIBvAAAAAElFTkSuQmCC)

# # Kaggle (5 баллов)
# 
# Как выставить баллы:
# 
# 1) 1 >= roc auc > 0.84 это 5 баллов
# 
# 2) 0.84 >= roc auc > 0.7 это 3 балла
# 
# 3) 0.7 >= roc auc > 0.6 это 1 балл
# 
# 4) 0.6 >= roc auc это 0 баллов
# 
# 
# Для выполнения задания необходимо выполнить следующие шаги.
# * Зарегистрироваться на платформе [kaggle.com](kaggle.com). Процесс выставления оценок будет проходить при подведении итогового рейтинга. Пожалуйста, укажите во вкладке Team -> Team name свои имя и фамилию в формате Имя_Фамилия (важно, чтобы имя и фамилия совпадали с данными на Stepik).
# * Обучить модель, получить файл с ответами в формате .csv и сдать его в конкурс. Пробуйте и экспериментируйте. Обратите внимание, что вы можете выполнять до 20 попыток сдачи на kaggle в день.
# * После окончания соревнования отправить в итоговый ноутбук с решением на степик. 
# * После дедлайна проверьте посылки других участников по критериям. Для этого надо зайти на степик, скачать их ноутбук и проверить скор в соревновании.
