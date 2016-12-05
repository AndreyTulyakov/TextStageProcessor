#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pymorphy2




import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

from TextData import TextData
from TextPreprocessing import *
from TextClasterization import *

# Для корректного отображение шрифтов на графиках в Windows
if(os.name != 'posix'):
    font = {'family': 'Verdana',
            'weight': 'normal'}
    rc('font', **font)

### ПРОГРАММА  ---------------------------------------------------------------

# Получаем экземпляр анализатора (10-20мб)
morph = pymorphy2.MorphAnalyzer()

# Сюда будем записывать текст выходных файлов
log_string = ""

# Загружаем предложения из нескольких файлов
texts = loadInputFiles("input_files")

# Разделяем предложения на слова
texts = tokenizeTextData(texts)


# Переводим обычное предложение в нормализованное (каждое слово)

log_string = "Нормализация:\n"
print('1) Нормализация.')

for text in texts:
    log_string = log_string + '\nText:' + text.filename + '\n'
    for sentence in text.tokenized_sentences:
        current_sentence = []
        for word in sentence:
            if(wordPersonDetector(word, morph) == False):
                result = morph.parse(word)[0] # По умолчанию берем наиболее достоверный разбора слова
                #current_sentence.append(word)
                current_sentence.append(result.normal_form)
                log_string = log_string + ' ' + result.normal_form
            else:
                current_sentence.append(word)
                log_string = log_string + ' ' + word
        log_string = log_string + '\n'
        text.normalized_sentences.append(current_sentence)

writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_1.txt')


# Удаление стоп-слов из предложения (частицы, прилагательные и тд)
for text in texts:
    text.no_stop_words_sentences = removeStopWordsFromSentences(text.normalized_sentences, morph)

log_string = "Удаление стоп-слов:\n"
print('2) Удаление стоп-слов.')
for text in texts:
    log_string = log_string + '\nText:' + text.filename + '\n'
    for sentence in text.no_stop_words_sentences:
        for word in sentence:
            log_string = log_string + ' ' + word
        log_string = log_string + '\n'

writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_2.txt')



# Приведение регистра (все слова с маленькой буквы за исключением ФИО)


log_string = "Приведение регистра:\n"
print('3) Приведение регистра.')

for text in texts:
    log_string = log_string + '\nText:' + text.filename + '\n'
    for sentence in text.no_stop_words_sentences:
        current_sentence = []
        for word in sentence:
            if(wordPersonDetector(word, morph) == True):
                current_sentence.append(word.capitalize())
            else:
                current_sentence.append(word.lower())
            log_string = log_string + ' ' + current_sentence[-1]
        text.register_pass_centences.append(current_sentence)
        log_string = log_string + '\n'

writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_3.txt')


# Подсчет частоты слов в тексте
# Сортируем слова по частоте
for text in texts:
    for sentense in text.register_pass_centences:
        for word in sentense:
            if word.isalpha():
                text.word_frequency[word] = text.word_frequency.get(word, 0) + 1

    text.sorted_word_frequency = sorted(text.word_frequency.items(), key= lambda x: x[1], reverse=True)


print('4) Расчет частотной таблицы слов.')
log_string = 'Расчет частотной таблицы слов.\n'

for text in texts:
    log_string = log_string + '\nText:' + text.filename + '\n'
    log_string = log_string + "Слово;Кол-во упоминаний\n"
    for key, value in text.sorted_word_frequency:
        log_string = log_string + key + ';' + str(value) + '\n' 
writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_4.csv')


# Удаляем редкие слова без поддержки (менее 2 слов)
# for text in texts:
#     word_frequency = text[6]
#     keys_for_delete = []
#     for key, value in word_frequency.items():
#         if(word_frequency[key] < 2):
#             keys_for_delete.append(key)

#     for del_key in keys_for_delete:
#         word_frequency.pop(del_key)


# print('5) Извлечение групп слов обладающих частотностью.')
# for text in texts:
#     word_groups = []
#     register_pass_centences = text[5]
#     for sentence in register_pass_centences:
#         word_groups.extend(formFrequencySet(word_frequency, sentence, 2, 2))

#     group_frequency = calculateGroupsSupport(word_groups)
#     text.append(group_frequency)
#     sorted_groups = sorted(group_frequency.items(), key= lambda x: x[1], reverse=True)
#     text.append(sorted_groups)

# log_string = "N-Граммы слов,Повторяемость N-Граммы.\n"
# # Сортируем словосочетания по частоте
# for text in texts:
#     sorted_groups = text[9]
#     for key, value in sorted_groups:
#         log_string = log_string + key + "," + str(value) + "\n"
# writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_5.csv')


print('6) Вычисление модели TF*IDF.')

idf_word_data = calculateWordsIDF(texts, )
sorted_IDF = sorted(idf_word_data.items(), key=lambda x: x[1], reverse=False)
calculateTFIDF(texts, idf_word_data)





log_string = writeWordTFIDFToString(texts, idf_word_data)
writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_6.csv')

# Вырезаем из TF-IDF 50% лучших слов
removeTFIDFWordsWithMiniamlMultiplier(texts , 0.25)

log_string = writeWordTFIDFToString(texts, idf_word_data)
writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_7.csv')




print("7) Латентно-семантический анализ")
def CreateLSAMatrix(texts):
    # Реализация латентно семантического анализа. LSA
    all_documents_count = len(texts);
    all_idf_word_list = dict()
    for text in texts:
        for word in list(text.words_tf_idf.keys()):
            if(idf_word_data.get(word, None) != None):
                all_idf_word_list[word] = idf_word_data[word]


    all_idf_word_keys = list(all_idf_word_list.keys())
    print("TOTAL WORDS:" + str(len(all_idf_word_list)))
    words_count = len(all_idf_word_keys)
    lsa_matrix = np.zeros(shape=(words_count,all_documents_count))

    for t in range(len(texts)):
        for i in range(len(all_idf_word_keys)):
            current_word = all_idf_word_keys[i]
            word_frequency_in_current_text = texts[t].word_frequency.get(current_word, 0)
            lsa_matrix[i][t] = math.sqrt(text.words_tf_idf.get(current_word,0.001)*word_frequency_in_current_text)

    return lsa_matrix, all_idf_word_keys


lsa_matrix, all_idf_word_keys = CreateLSAMatrix(texts)
#print(lsa_matrix)

def divideSingular(matrix):
    u,s,v = np.linalg.svd(lsa_matrix, True)
    S = np.zeros((u.shape[0], v.shape[0]), dtype=complex)
    S[:v.shape[0], :v.shape[1]] = np.diag(s)
    return u, S, v, s

u, S, v, s = divideSingular(lsa_matrix)


minimal_singular_value = np.max(s)*0.5
minimal_row_index = 0
for i in range(len(s)):
    if(s[i] > minimal_singular_value):
        minimal_row_index = i

minimal_row_index = 2

nu = u[0:,0:(minimal_row_index+1)]
ns = S[0:(minimal_row_index+1),0:(minimal_row_index+1)]
nv = v[0:(minimal_row_index+1),0:]

# Создание матрицы СЛОВО / ЧастотаСловаВ(Док1), ЧастотаСловаВ(Док1), ...

plt.plot(nu[0],nu[1],'go') #Построение графика
plt.plot(nv[0],nv[1],'go') #Построение графика

plt.xlabel('x') #Метка по оси x в формате TeX
plt.ylabel('y') #Метка по оси y в формате TeX
plt.title('LSA') #Заголовок в формате TeX
plt.grid(True) #Сетка

#====================================================================================
min_value = 0.1
divider = 1
#need_words = True
need_words = False


if(need_words):
    for i in range(int(nu.shape[0])):
        if(abs(nu[i][0])>min_value or abs(nu[i][1])>min_value or abs(nu[i][2])>min_value ):
            plt.annotate(str(all_idf_word_keys[i]), xy=(nu[i][0],nu[i][1]), textcoords='data')

for i in range(len(texts)):
        plt.annotate(str(texts[i].filename), xy=(nv[0][i],nv[1][i]), textcoords='data')


plt.show() #Показать график

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
nu = np.transpose(nu)


nuxx = []
nuxy = []
nuxz = []
for i in range(len(nu[0])):
    if(abs(nu[0][i])>min_value or abs(nu[1][i])>min_value or abs(nu[2][i])>min_value):
        nuxx.append(nu[0][i])
        nuxy.append(nu[1][i])
        nuxz.append(nu[2][i])


if(need_words):
   ax.scatter(nuxx,nuxy,nuxz, c='r')#, marker='o')
ax.scatter(nv[0],nv[1],nv[2], c='b', marker='^')


for i in range(len(texts)):
       ax.text(nv[0][i], nv[1][i], nv[2][i], str(texts[i].filename), None)

if(need_words):
   for i in range(len(nuxx)):
       if(abs(nu[0][i])>min_value or abs(nu[1][i])>min_value or abs(nu[2][i])>min_value):
           ax.text(nuxx[i],nuxy[i],nuxz[i], str(all_idf_word_keys[i]), None)


plt.show() #Показать график
