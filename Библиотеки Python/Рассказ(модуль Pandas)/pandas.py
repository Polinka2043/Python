#!/usr/bin/env python
# coding: utf-8

# ### Task 2

# In[1]:


import pandas as pd


# In[2]:


series = pd.Series (["Toyota","Ford","BMW","Audi","Mercedes","Honda","Tesla","Volvo","Lexus","Porsche"])


# In[3]:


series


# In[4]:


df = pd.DataFrame({"brand": ["Camry","Mustang","X5","Q7","GLE","Accord","Model S","XC90","RX","Macan"]}, index = ["Toyota","Ford","BMW","Audi","Mercedes","Honda","Tesla","Volvo","Lexus","Porsche"])


# In[5]:


df


# In[6]:


df = pd.read_csv("1.csv", delimiter=";", encoding="latin1")


# In[7]:


df


# In[8]:


df.shape


# In[9]:


df.head()


# In[10]:


df.tail()


# ###  Task 3

# In[11]:


import pandas as pd


# In[12]:


df = pd.read_csv('work3.csv', delimiter=";")


# In[13]:


df


# In[14]:


# 1. Вывести только фильмы с определённым жанром 
movies = df[df['Жанр'] == 'комедия']


# In[15]:


movies


# In[16]:


# 2. Вывести только фильмы, которые были сняты после 2010 года
movies_after_2010 = df[df['Год'] > 2010]


# In[17]:


movies_after_2010


# In[18]:


# 3. Вывести только фильмы с оценкой выше 4.0
high_rated_movies = df[df['Средняя оценка'] > 4.0]


# In[19]:


high_rated_movies


# In[20]:


# 4. Вывести только фильмы, у которых больше 10 000 оценок
popular_movies = df[df['Количество оценок'] > 10000]


# In[21]:


popular_movies


# ###  Task 4

# #### Линейная диаграмма:

# In[22]:


import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y)
plt.show()


# #### Столбчатая диаграмма:

# In[23]:


x = ['A', 'B', 'C', 'D', 'E']
y = [3, 7, 2, 5, 8]

plt.bar(x, y)
plt.show()


# #### Гистограмма:

# In[24]:


data = [1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

plt.hist(data, bins=5)
plt.show()


# #### Круговая диаграмма:

# In[25]:


labels = ['A', 'B', 'C', 'D']
sizes = [3, 4, 6, 2]

plt.pie(sizes, labels=labels)
plt.show()


# #### Диаграмма рассеяния:

# In[26]:


x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.scatter(x, y)
plt.show()


# ### Task 5

# In[27]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('work3.csv', delimiter=";")


# In[28]:


# 1. Построить график изменения средней оценки фильмов из года в год. Разделение сделать по десятилетиям.
df['Десятилетие'] = ((df['Год'] // 10) * 10)
average_ratings = df.groupby('Десятилетие')['Средняя оценка'].mean()
average_ratings.plot(kind='line')
plt.show()


# In[29]:


# 2. Построить столбчатую диаграмму разделения ваших любимых фильмов по жанрам: сколько каких фильмов какого жанра у вас представлено в таблице.
genre_counts = df['Жанр'].value_counts()
genre_counts.plot(kind='bar')
plt.show()


# In[30]:


# 3. Отсортировать фильмы в порядке уменьшения средней оценки. Вывести столбчатую диаграмму.
sorted_movies = df.sort_values(by='Средняя оценка', ascending=False)
plt.bar(sorted_movies['Название'], sorted_movies['Средняя оценка'])
plt.xticks(rotation=90)
plt.show()


# ### Task 6

# In[31]:


import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('work3.csv', delimiter=";")

fig, axs = plt.subplots(3, figsize=(10, 15))

# Построить график изменения средней оценки фильмов из года в год. Разделение сделать по десятилетиям.
df['Десятилетие'] = (df['Год'] // 10) * 10
average_ratings = df.groupby('Десятилетие')['Средняя оценка'].mean()
average_ratings.plot(kind='line', ax=axs[0])
axs[0].set_title('Средняя оценка по десятилетиям')

# Построить столбчатую диаграмму разделения ваших любимых фильмов по жанрам: сколько каких фильмов какого жанра у вас представлено в таблице.
genre_counts = df['Жанр'].value_counts()
genre_counts.plot(kind='bar', ax=axs[1])
axs[1].set_title('Распределение фильмов по жанрам')

# Отсортировать фильмы в порядке уменьшения средней оценки. Вывести столбчатую диаграмму.
sorted_movies = df.sort_values(by='Средняя оценка', ascending=False)
axs[2].bar(sorted_movies['Название'], sorted_movies['Средняя оценка'])
axs[2].set_title('Фильмы, отсортированные по средней оценке')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# ### Task 7

# ### Структура данных: Реляционные базы данных используют структурированные     таблицы с фиксированными схемами, в то время как нереляционные базы                     данных, такие как MongoDB, могут хранить данные в более гибких структурах,               таких как документы, пары ключ-значение, графы или широкие столбцы.
# ### Связи между данными: В реляционных базах данных данные связаны между собой через ключи, что позволяет выполнять сложные запросы с использованием операций соединения. Нереляционные базы данных обычно не поддерживают операции соединения, и связи между данными обычно устанавливаются на уровне приложения.
# ### Масштабируемость: Реляционные базы данных обычно масштабируются вертикально путем увеличения мощности одного сервера, в то время как нереляционные базы данных, такие как MongoDB, обычно масштабируются горизонтально путем добавления серверов в пул данных.
# ### Согласованность данных: Реляционные базы данных обычно обеспечивают строгую согласованность данных, следуя принципам ACID (атомарность, согласованность, изолированность, долговечность). Нереляционные базы данных часто предлагают более гибкие модели согласованности, которые могут быть настроены для удовлетворения требований конкретного приложения.

# ### Task 8

# In[32]:


from pymongo import MongoClient

# Создать подключение к серверу MongoDB
client = MongoClient('localhost', 27017)

# 1. Создать базу данных
db = client.synergy

# 2. Создать коллекцию
collection = db.synergy

# 3. Добавить в коллекцию информацию о ваших любимых фильмах
movies = [
    {"Название": "Король улиц", "Жанр": "комедия", "Год": 2023, "Количество оценок": 2472, "Средняя оценка": 6.2, "Длительность": 106},
    {"Название": "Транзит", "Жанр": "боевик", "Год": 2010, "Количество оценок": 3760, "Средняя оценка": 5.5, "Длительность": 99},
    {"Название": "Остров проклятых", "Жанр": "триллер", "Год": 2009, "Количество оценок": 899340, "Средняя оценка": 8.5, "Длительность": 149},
    {"Название": "Оборотень", "Жанр": "боевик", "Год": 2013, "Количество оценок": 5932, "Средняя оценка": 5.1, "Длительность": 107},
    {"Название": "На линии огня", "Жанр": "триллер", "Год": 2010, "Количество оценок": 9369, "Средняя оценка": 6.3, "Длительность": 134},
    {"Название": "Спартанец", "Жанр": "драма", "Год": 2004, "Количество оценок": 6794, "Средняя оценка": 6.2, "Длительность": 177},
    {"Название": "Загнанный", "Жанр": "криминал", "Год": 2003, "Количество оценок": 20380, "Средняя оценка": 6.8, "Длительность": 163},
    {"Название": "Боксер", "Жанр": "спорт", "Год": 2010, "Количество оценок": 3356, "Средняя оценка": 6.1, "Длительность": 140},
    {"Название": "Защитник", "Жанр": "триллер", "Год": 2021, "Количество оценок": 1161, "Средняя оценка": 6.0, "Длительность": 97},
    {"Название": "Гонщики на драйве", "Жанр": "боевик", "Год": 2023, "Количество оценок": 15412, "Средняя оценка": 6.6, "Длительность": 125}
]
collection.insert_many(movies)

# 4. Получить информацию о фильмах, которые вышли в 2010 году
movies_2010 = collection.find({"Год": 2010})

# Вывести информацию о фильмах
for movie in movies_2010:
    print(movie)

# 5. Вывести на экран среднюю оценку этих фильмов
ratings = [movie['Средняя оценка'] for movie in collection.find({"Год": 2010})]
average_rating = sum(ratings) / len(ratings) if ratings else 0
print("Средняя оценка фильмов 2010 года:", average_rating)


# ### Task 9-11

# In[33]:


import os
import time
from pymystem3 import Mystem

# Создаем экземпляр Mystem
m = Mystem()

# Загружаем текст
with open('prestuplenie-i-nakazanie.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Замеряем время работы
start_time = time.time()

# Применяем лемматизацию
lemmas = m.lemmatize(text)

# Замеряем время окончания работы
end_time = time.time()

# Сохраняем результат в файл
with open('lemmatized.txt', 'w', encoding='utf-8') as f:
    for lemma in lemmas:
        f.write(lemma)

# Выводим затраченное время
print('Время работы: ', end_time - start_time)


# In[34]:


import json
import nltk
nltk.download('punkt')
import pymorphy3
import time
from nltk.tokenize import word_tokenize

# Инициализация Pymorphy
morph = pymorphy3.MorphAnalyzer()

# Загрузка текста книги
with open('prestuplenie-i-nakazanie.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Токенизация текста
tokens = word_tokenize(text)

# Разбор слов
parsed_words = []
start_time = time.time()
for token in tokens:
    parsed_word = morph.parse(token)[0]
    parsed_words.append({
        'lemma': parsed_word.normal_form,
        'word': parsed_word.word,
        'pos': parsed_word.tag.POS
    })
end_time = time.time()

# Сохранение результатов в jsonlines
with open('parsed_words.jsonl', 'w', encoding='utf-8') as f:
    for word in parsed_words:
        f.write(json.dumps(word, ensure_ascii=False) + '\n')

print(f'Время работы: {end_time - start_time} секунд')


# In[35]:


import json
import nltk
from collections import Counter
from nltk.util import ngrams
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Создаем экземпляр MorphAnalyzer
morph = MorphAnalyzer()

# Загружаем текст
with open('prestuplenie-i-nakazanie.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Токенизируем текст
tokens = word_tokenize(text)

# Разбираем слова с помощью pymorphy и считаем части речи
pos_counter = Counter()
word_counter = Counter()
for token in tokens:
    parsed_token = morph.parse(token)[0]
    pos_counter[parsed_token.tag.POS] += 1
    if parsed_token.tag.POS in ['VERB', 'ADVB']:
        word_counter[parsed_token.normal_form] += 1

# Выводим долю слов каждой части речи
total_words = sum(pos_counter.values())
for pos, count in pos_counter.items():
    print(f'{pos}: {count / total_words:.2%}')

# Выводим топ-20 глаголов и наречий
for word, count in word_counter.most_common(20):
    print(f'{word}: {count}')

# Создаем биграммы и триграммы
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# Выводим топ-25 биграмм и триграмм
print(Counter(bigrams).most_common(25))
print(Counter(trigrams).most_common(25))


# In[36]:


from pymorphy3 import MorphAnalyzer
from nltk.tokenize import word_tokenize

# Создаем экземпляр MorphAnalyzer
morph = MorphAnalyzer()

# Загружаем текст
text = 'Он благополучно избегнул встречи с своею хозяйкой на лестнице. Каморка его приходилась под самою кровлей высокого пятиэтажного дома и походила более на шкаф, чем на квартиру. Квартирная же хозяйка его, у которой он нанимал эту каморку с обедом и прислугой, помещалась одною лестницей ниже, в отдельной квартире, и каждый раз, при выходе на улицу, ему непременно надо было проходить мимо хозяйкиной кухни, почти всегда настежь отворенной на лестницу. И каждый раз молодой человек, проходя мимо, чувствовал какое-то болезненное и трусливое ощущение, которого стыдился и от которого морщился. Он был должен кругом хозяйке и боялся с нею встретиться.'

# Токенизируем текст
tokens = word_tokenize(text)

# Изменяем время глаголов и число существительных
new_tokens = []
for token in tokens:
    parsed_token = morph.parse(token)[0]
    if parsed_token.tag.POS == 'VERB':
        # Изменяем время глагола на будущее, если это возможно
        inflected_token = parsed_token.inflect({'futr'})
        new_token = inflected_token.word if inflected_token is not None else token
    elif parsed_token.tag.POS == 'NOUN':
        # Изменяем число существительного на множественное, если это возможно
        inflected_token = parsed_token.inflect({'plur'})
        new_token = inflected_token.word if inflected_token is not None else token
    else:
        new_token = token
    new_tokens.append(new_token)

# Соединяем токены обратно в текст
new_text = ' '.join(new_tokens)
print(new_text)

