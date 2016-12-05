#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2

import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

from TextData import TextData
from TextPreprocessing import *
from TextClasterization import *
from TextLSA import *

# Для корректного отображение шрифтов на графиках в Windows
if(os.name != 'posix'):
    font = {'family': 'Verdana',
            'weight': 'normal'}
    rc('font', **font)

### ПРОГРАММА  ---------------------------------------------------------------

# Получаем экземпляр анализатора (10-20мб)
morph = pymorphy2.MorphAnalyzer()

# Загружаем предложения из нескольких файлов
texts = loadInputFiles("input_files")

# Разделяем предложения на слова
texts = tokenizeTextData(texts)

# Удаление стоп-слов из предложения (частицы, прилагательные и тд)
print('1) Удаление стоп-слов.')
texts, log_string = removeStopWordsInTexts(texts, morph)
writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_1.txt')

# Переводим обычное предложение в нормализованное (каждое слово)
print('2) Нормализация.')
texts, log_string = normalizeTexts(texts, morph)
writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_2.txt')

# Приведение регистра (все слова с маленькой буквы за исключением ФИО)
print('3) Приведение регистра.')
texts, log_string = fixRegisterInTexts(texts, morph)
writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_3.txt')

# Подсчет частоты слов в тексте
print('4) Расчет частотной таблицы слов.')
texts, log_string = calculateWordsFrequencyInTexts(texts)
writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_4.csv')


print('6) Вычисление модели TF*IDF.')
idf_word_data = calculateWordsIDF(texts, )
sorted_IDF = sorted(idf_word_data.items(), key=lambda x: x[1], reverse=False)
calculateTFIDF(texts, idf_word_data)

log_string = writeWordTFIDFToString(texts, idf_word_data)
writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_6.csv')

# Вырезаем из TF-IDF % худших слов
removeTFIDFWordsWithMiniamlMultiplier(texts , 0.05)

log_string = writeWordTFIDFToString(texts, idf_word_data)
writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_7.csv')


print("7) Латентно-семантический анализ")
lsa_matrix, all_idf_word_keys = CreateLSAMatrix(texts, idf_word_data)
u, S, v, s = divideSingular(lsa_matrix)
nu, ns, nv = cutSingularValue(u, S, v, s)

# Создание матрицы СЛОВО / ЧастотаСловаВ(Док1), ЧастотаСловаВ(Док1), ...
need_words = True
#need_words = False

viewLSAGraphics2D(plt, nu, nv, need_words, all_idf_word_keys, texts)
viewLSAGraphics3D(plt, nu, nv, need_words, all_idf_word_keys, texts)


