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

# In[4]:


get_ipython().run_cell_magic('capture', '', '%%bash\npip install catboost')


# In[52]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LogisticRegressionCV
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from catboost import CatBoostClassifier, Pool, cv as catboost_cv
from sortedcontainers import SortedList
import copy
import collections
from itertools import product,chain
import random
random_state = 42
random.seed(42)
np.random.seed(42)
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# In[178]:


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

# In[80]:


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


# In[81]:


data = pd.read_csv('https://drive.google.com/uc?id=1flwlI7XSSwpqJ5tz-lvNyE70BIz7McnL')


# In[82]:


data = shuffle(data, random_state=random_state)


# In[83]:


data.info()


# In[84]:


data.head()


# In[85]:


describe_full(data)


# In[86]:


pd.DataFrame({"col":data.select_dtypes(include="object").columns,
              "unique": [len(data[col].unique()) for col in data.select_dtypes(include="object").columns],
              "values": [data[col].unique() for col in data.select_dtypes(include="object").columns]})


# In[87]:


try:
  pd.to_numeric(data["TotalSpent"], errors="raise")
except Exception as e:
  print(e)


# In[88]:


data["TotalSpent"] = data["TotalSpent"].str.strip()
data.loc[data["TotalSpent"] == "", "TotalSpent"] = "0.0" 
data["TotalSpent"] = pd.to_numeric(data["TotalSpent"], errors="raise")


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


# In[109]:


for col in num_cols:
  plt.hist(data[col])
  plt.title(col)
  plt.show()


# In[100]:


sns.boxplot(data['TotalSpent'])


# наиболее важные фичи по мнению катбуст

# In[101]:


X, y = data[feature_cols], data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
train_ds = Pool(data=X_train, label=y_train, cat_features=cat_cols, feature_names=feature_cols)
test_ds = Pool(data=X_test, label=y_test, cat_features=cat_cols, feature_names=feature_cols)
full_ds = Pool(data=X, label=y, cat_features=cat_cols, feature_names=feature_cols)
scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
cb = CatBoostClassifier(verbose=0, scale_pos_weight=scale_pos_weight,task_type="GPU", 
                        devices='0:1', random_seed=random_state).fit(train_ds)
df_feature_importances = pd.DataFrame(((zip(cb.feature_names_, cb.get_feature_importance())))).rename(columns={0:"feature",1:"coeff"}).sort_values(by="coeff", ascending = False )
sns.barplot(data=df_feature_importances, x=df_feature_importances["coeff"], y=df_feature_importances["feature"])


# In[102]:


del X, y, X_train, X_test, y_train, y_test, train_ds, test_ds, full_ds


# In[103]:


sns.countplot(data["Churn"])


# In[118]:


fig, axs = plt.subplots(4, len(cat_cols) // 4, figsize=(15,15))
axs = axs.flatten()
plt.tight_layout()
for cat_col, ax in zip(cat_cols, axs):
     data[cat_col].value_counts().plot(kind='bar', ax=ax)
     ax.legend()

plt.show()


# In[104]:


display(data["Churn"].value_counts())
100*data["Churn"].value_counts()/len(data)


# как видим классы несбалансированы, неплохо бы оверсэмплить(либо указывать веса катабусту)

# In[181]:


fig, axs = plt.subplots(4, len(cat_cols) // 4, figsize=(15,15))
axs = axs.flatten()

for cat_col, ax in zip(cat_cols, axs):
     display_group_density_plot(data, groupby = cat_col, on = target_col,                                            palette = sns.color_palette('Set2'), 
                                title=cat_col,
                           figsize = (10, 5), ax=ax)

plt.tight_layout()
plt.show()


# In[186]:


fig, axs = plt.subplots(3, len(num_cols) // 3, figsize=(15,15))
axs = axs.flatten()

for num_col, ax in zip(num_cols, axs):
     display_group_density_plot(data, groupby = num_col, on = target_col,                                            palette = sns.color_palette('Set2'), 
                                title=num_col,
                           figsize = (10, 5), ax=ax)

plt.tight_layout()
plt.show()


# In[41]:


sns.distplot(data["TotalSpent"])
plt.axvline(0, c="r", label="")
plt.legend()


# (Дополнительно) Если вы нашли какие-то ошибки в данных или выбросы, то можете их убрать. Тут можно поэксперементировать с обработкой данных как угодно, но не за баллы.

# числовые данные плюс минус в порядке, TotalSpent можно отскалировать, категориальные не смотрел

# In[187]:


for col in num_cols:
  sns.boxplot(data[col])
  plt.title(col)
  plt.show()


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

# In[188]:


df = data.copy()


# In[189]:


for col in ["HasPartner",	"HasChild", "HasPhoneService", "IsBillingPaperless"]:
  df[col] = (df[col] == "Yes").astype("int")
df["Sex"] = (df["Sex"] == "Male").astype("int")


# In[192]:


df = pd.get_dummies(df, drop_first=True)


# In[197]:


features = set(df.columns) - set([target_col])
X, y = df[features].values, df[target_col].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
lr = LogisticRegressionCV(cv=5, random_state=42).fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[198]:


display_classification_report(y_test, y_pred)


# In[199]:


fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
plot_pr(y_test, y_pred, ax=ax[0],label="LogisticRegressionCV")
plot_roc(y_test, y_pred, ax=ax[1],label="LogisticRegressionCV")


# In[200]:


roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])


# In[201]:


lrcv = LogisticRegressionCV(
    Cs=[0.1,1,10], penalty='l2', tol=1e-10, scoring='neg_log_loss', cv=5,
    solver='liblinear', n_jobs=4, verbose=0, refit=True,
    max_iter=100,
).fit(X_train, y_train)
y_pred = lrcv.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="LogisticRegressionCV")
plot_roc(y_test, y_pred, ax=axs[1], label="LogisticRegressionCV")


# Выпишите какое лучшее качество и с какими параметрами вам удалось получить

# 

# In[202]:


params = {
"criterion":["gini", "entropy"],
"max_depth":[2,4,8,16],
"min_samples_split":[2,4,8, 16],
"min_samples_leaf":[2,4,6]}
clf = GridSearchCV(DecisionTreeClassifier(), params, cv=5).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(clf.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)


# In[204]:


roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])


# In[206]:


dt = DecisionTreeClassifier(**clf.best_params_).fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(dt.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="DecisionTreeClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="DecisionTreeClassifier")


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

# In[269]:


X, y = data[feature_cols], data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
train_ds = Pool(data=X_train, label=y_train, cat_features=cat_cols, feature_names=feature_cols)
test_ds = Pool(data=X_test, label=y_test, cat_features=cat_cols, feature_names=feature_cols)
full_ds = Pool(data=X, label=y, cat_features=cat_cols, feature_names=feature_cols)


# In[208]:


scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()


# In[209]:


cb = CatBoostClassifier(verbose=0, scale_pos_weight=scale_pos_weight,task_type="GPU", 
                        devices='0:1', random_seed=random_state).fit(train_ds)


# In[55]:


df_feature_importances = pd.DataFrame(((zip(cb.feature_names_, cb.get_feature_importance())))).rename(columns={0:"feature",1:"coeff"}).sort_values(by="coeff", ascending = False )
sns.barplot(data=df_feature_importances, x=df_feature_importances["coeff"], y=df_feature_importances["feature"])


# In[210]:


y_pred = cb.predict(test_ds)
print(cb.score(train_ds))


# In[211]:


print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="CatBoostClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="CatBoostClassifier")


# In[217]:


param_grid = {
        'learning_rate': [0.03, 0.1],
        'depth': [6, 10, 20, 40],
        'l2_leaf_reg': [3, 5, 7, 9],
        'iterations':[20, 60],
        'thread_count':[12],
        'border_count':[128]
}

model = CatBoostClassifier(loss_function='Logloss',eval_metric='AUC', task_type="GPU", devices='0:1', random_seed=random_state)
grid_search_result = model.grid_search(param_grid, 
                                       full_ds,
                                       verbose=0,
                                       partition_random_seed=random_state,
                                       search_by_train_test_split=True,
                                       train_size=0.9,
                                       plot=False)


# In[220]:


cv_data = pd.DataFrame(grid_search_result["cv_results"])
best_value = cv_data['test-AUC-mean'].max()
best_iter = cv_data['test-AUC-mean'].values.argmax()

print('Best validation test-AUC-mean : {:.4f}±{:.4f} on step {}'.format(
    best_value,
    cv_data['test-AUC-std'][best_iter],
    best_iter)
)


# In[227]:


model = CatBoostClassifier(loss_function='Logloss', task_type="GPU", devices='0:1', random_seed=random_state, **grid_search_result["params"])
model.fit(train_ds, verbose = 0, eval_set = [(X_test, y_test)], use_best_model=True)
y_pred = model.predict(test_ds)
print("accuracy_score", accuracy_score(y_test, y_pred))
for i in [10, 15, 20]:
  print("roc_auc_score", roc_auc_score(y_test, model.predict_proba(test_ds, ntree_start=0, ntree_end=i)[:,1]))
print("f1_score", f1_score(y_test, y_pred))


# In[230]:


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
        


# In[ ]:


import pandas
import numpy as np
import catboost as cb
from sklearn.model_selection import KFold
from itertools import product,chain

train_set = X_train
test_set = X_test

# X, y = data[feature_cols], data[target_col]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
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

params = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[250,100,500,1000],
          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20,50,100,200],
          'thread_count':4}

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

bestparams = catboost_param_tune(params,train_set,train_label,cat_dims)


# In[279]:


X, y = data[feature_cols], data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
train_ds = Pool(data=X_train, label=y_train, cat_features=cat_cols, feature_names=feature_cols)
test_ds = Pool(data=X_test, label=y_test, cat_features=cat_cols, feature_names=feature_cols)
full_ds = Pool(data=X, label=y, cat_features=cat_cols, feature_names=feature_cols)

clf = CatBoostClassifier(**bestparams, verbose=0)
clf.fit(train_ds)
roc_auc_score(y_test, clf.predict_proba(test_ds)[:,1])


# In[257]:


bestparams = {'depth': 4, 'iterations': 500, 'learning_rate': 0.03, 'l2_leaf_reg': 3, 'border_count': 10, 'thread_count': 4}
clf = CatBoostClassifier(**bestparams, verbose=0)
clf.fit(train_set, np.ravel(train_label), cat_features=cat_dims)
roc_auc_score(test_label, clf.predict_proba(test_set)[:,1])


# Выпишите какое лучшее качество и с какими параметрами вам удалось получить

# 

# # Предсказания

# In[289]:


best_model = clf


# In[291]:


X_test = pd.read_csv('https://drive.google.com/uc?id=1JmoeWw0Y3M7q5DunTVmWioMgXDXBt02r')
test_ds = Pool(data=X_test.values, cat_features=cat_cols, feature_names=feature_cols)


# In[297]:


submission = pd.read_csv('https://drive.google.com/uc?id=15lHpRREzDVihgDZ866CXueuibKEgPRJg')

submission['Churn'] = clf.predict_proba(test_ds)[:,1] # best_model.predict_proba(X_test) / best_model.predict(X_test)
submission.to_csv('./my_submission.csv', index=False)


# In[298]:


##colab
try:
  from google.colab import files
  files.download('my_submission.csv')
except:
  pass


# результат 0.85094 ,что достаточно (поскольку больше 84;))
# не нашел где взять номер посылки, скрин ниже

# ![image.png](data:image/png;base64,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XS8i0z6lUdo0uxZTXtrxbON8hQEYgIGVzQDiLeJt6gSCE39ln/j0L/0LAzAAAzAAAzAAAzAAA9kMJEMoSMBVSIOFDGOgvJJhElTOQpZBH2f3MbbBNjAAA8uFAcRbxFvEWxiAARiAARiAARiAARiAARiAgQQDCmWQJq5qm7xlk/FwCxEBdIyOzco3LURDIfmSBhEKBmAABlYuA4i3iQs0sK9c2Olb+hYGYAAGYAAGYAAGYAAGYKBYBk4PDJs8YtP+JMJqv7xlJbzGgq7WtU37lCZNsPU8tb/YepEelmEABmBgdTCAeIt4yyQBBmAABmAABmAABmAABmAABmAgDwMSYrvyiLguwha7VJ6x4IsQszqEGPqZfoYBGCiGAcTbPBfoYgxJWk48GIABGIABGIABGIABGIABGFjZDEhozQp7UKhweyFhF+BrZfNF/9K/MAADaQwg3iLe8oQdBmAABmAABmAABmAABmAABmCgSAYk5CokgrxnJcieGRiy3ii8gta1TfuURmnxskWYSRNm2AYXMAAD+RhAvC3yAp3PmOzjZIMBGIABGIABGIABGIABGIABGIABGIABGIABGFgoBhBvEW95wg4DMAADMAADMAADMAADMAADMAADMAADMAADMFCCDCDelmCnLJQyTz485YEBGIABGIABGIABGIABGIABGIABGIABGICB5csA4i3iLU9VYAAGYAAGYAAGYAAGYAAGYAAGYAAGYAAGYAAGSpABxNsS7BSehizfpyH0HX0HAzAAAzAAAzAAAzAAAzAAAzAAAzAAAzCwUAwsmXgbF8T6RsMG2AAGYAAGYAAGYAAGYAAGYAAGYAAGYAAGYAAGYKBYBiz6vCdav6DVYitBesCFARiAARiAARiAARiAARiAARiAARiAARiAARiAgZkMxCIt4u0dM40DLNgDBmAABmAABmAABmAABmAABmAABmAABmAABmDgYjGAeItgS1gHGIABGIABGIABGIABGIABGIABGIABGIABGICBEmQA8bYEO+ViKfmUy1MkGIABGIABGIABGIABGIABGIABGIABGIABGCgdBhBvEW95qgIDMAADMAADMAADMAADMAADMAADMAADMAADMFCCDCDelmCn8HSjdJ5u0Bf0BQzAAAzAAAzAAAzAAAzAAAzAAAzAAAzAwMViAPEW8ZanKjAAAzAAAzAAAzAAAzAA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ARmQARmQARmQgYUwoPF2IRXhzOrhZy6UuTKXARmQARmQARmQARmQARmQARmQARmQARlYMgMabzXeOpMiAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzKwQAY03i6wUpZs7TdvzkbJgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIwGEY0Hir8dZZFRmQARmQARmQARmQARmQARmQARmQARmQARmQgQUyoPF2gZXizMVhZi6Us3KWARmQARmQARmQARmQARmQARmQARmQARlYMgMabzXeOqsiAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzKwQAY03i6wUpZs7TdvzkbJgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIwGEY0Hir8dZZFRmQARmQARmQARmQARmQARmQARmQARmQARmQgQUyoPF2gZXizMVhZi6Us3KWARmQARmQARmQARmQARmQARmQARmQARlYMgMabzXeOqsiAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzKwQAY03i6wUpZs7TdvzkbJgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIwGEY0Hir8dZZFRmQARmQARmQARmQARmQARmQARmQARmQARmQgQUyoPF2gZXizMVhZi6Us3KWARmQARmQARmQARmQARmQARmQARmQARlYMgMabzXeOqsiAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzKwQAY03i6wUpZs7TdvzkbJgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIwGEY0Hir8dZZFRmQARmQARmQARmQARmQARmQARmQARmQARmQgQUyoPF2gZXizMVhZi6Us3KWARmQARmQARmQARmQARmQARmQARmQARlYMgMabzXeOqsiAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzKwQAY03i6wUpZs7TdvzkbJgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIwGEY0Hir8dZZFRmQARmQARmQARmQARmQARmQARmQARmQARmQgQUyoPF2gZXizMVhZi6Us3KWARmQARmQARmQARmQARmQARmQARmQARlYMgMabzXeOqsiAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzKwQAY03i6wUpZs7TdvzkbJgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIwGEY2Ivxlkh/9G/XatyeKwEloASUgBJQAkpACSgBJaAElIASUAJKQAkoASWgBJTABgkM2VV/akO4WbeHEpkViZ6VgBJQAkpACSgBJaAElIASUAJKQAkoASWgBJSAElACJySBIbuqxtsTgsCiKgEloASUgBJQAkpACSgBJaAElIASUAJKQAkoASWwPAlovF1enZgjJaAElIASUAJKQAkoASWgBJSAElACSkAJKAEloASUwOB2tK68FQ4loASUgBJQAkpACSgBJaAElIASUAJKQAkoASWgBJTABUrAlbcXKHyTVgJKQAkoASWgBJSAElACSkAJKAEloASUgBJQAkpACQxJYMh4+3/KZSyAcY0aUgAAAABJRU5ErkJggg==)

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
