#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs

import pymorphy2
import math
from pymorphy2 import tokenizers 
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
# Считывает имена всех .txt файлов из входной директории

# Для корректного отображение шрифтов на графиках в Windows
font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)

input_dir_name = "input_files"

input_filenames = []
for filename in os.listdir(input_dir_name):
    if filename.endswith(".txt"):
        input_filenames.append(filename);


# Основная структура данных где хранятся промежуточные варианты по тексту
class TextData:
    def __init__(self, filename):
        # Имя файла с текстом
        self.filename = filename
        # Исходные предложения не разделенные на слова
        self.original_sentences = []
        # Разделенные по словам предложения
        self.tokenized_sentences = []
        # Нормализованные предложения
        self.normalized_sentences = []
        # Предложения без стоп-слов
        self.no_stop_words_sentences = []
        # Предложения с преобразованным регистром (заглавные буквы только для имён)
        self.register_pass_centences = []
        # Словарь [слово:частота] по всему тексту
        self.word_frequency = dict()
        # Сортированный список [слово:частота]
        self.sorted_word_frequency = []
        # Счетчик слов в тексте
        self.word_count = 0
        # Рассчитанные веса TF*IDF для каждого слова в тексте (с учетом соседних документов)
        self.words_tf_idf = dict()


### ФУНКЦИИ ------------------------------------------------------------------

# Читает текст из файла и возвращает список предложений (без запятых)
def readSentencesFromInputText(filename):

    with codecs.open(input_dir_name + '/' + filename, 'r', "utf-8") as text_file:
        data=text_file.read().replace('\n', ' ')
        sentences = data.split('.')
        for i in range(len(sentences)):
            sentences[i] = sentences[i].strip().replace(',', '')
        return sentences
    return None

# Записывает/перезаписывает строку любой длины c переносами (data_str) в файл (filename)
def writeStringToFile(data_str, filename):
    with open(filename, 'w') as out_text_file:
        out_text_file.write(data_str)

# Определяет является ли слово частью ФИО (с вероятностью score)
def wordPersonDetector(word, morph):
    results = morph.parse(word)
    for result in results:
        if('Name' in result.tag
                or 'Surn' in result.tag
                or 'Patr' in result.tag):
            if(result.score >= 0.05):
                return True
    return False

# Удаляет СТОП-СЛОВа (Предлоги, союзы и тд.)
def removeStopWordsFromSentences(sentences, morph):
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            current_word = sentence[i]

            if(len(current_word) < 3):
                 sentence.pop(i)
                 i = i - 1
            else:
                results = morph.parse(current_word)
                for result in results:
                    if(result.tag.POS == 'ADJF'
                        or result.tag.POS == 'ADJS'
                        or result.tag.POS == 'PREP'
                        or result.tag.POS == 'ADVB'
                        or result.tag.POS == 'COMP'
                        or result.tag.POS == 'CONJ'
                        or result.tag.POS == 'PRCL'):
                        if(result.score >= 0.25):
                            sentence.pop(i)
                            i = i - 1
                            break
            i = i + 1
    return sentences

# Проверяет является ли слово местоимением-существительным (Он, Она и тд.)
def wordPersonNPRODetector(word, morph):
    results = morph.parse(word)
    for result in results:
        if(result.tag.POS == 'NPRO'):
            if(result.score >= 0.2):
                return True
    return False

# Заменяет ОН, ОНА и тд. на последнюю упомянутую персону в предложении.
def deanonimizeSentence(updated_sentence, deanon_stack, morph):
    result_sentence = []
    for word in updated_sentence:
        if(wordPersonNPRODetector(word, morph) and len(deanon_stack)>0):
            result_sentence.append(deanon_stack[-1][0])
        else:
            result_sentence.append(word)

    return result_sentence


# Извлекает из текста наиболее большие группы слов из sentence в которых
# поддержка каждого слова не менее чем minimalSupport, а группа не менее чем из minimalSize слов
def formFrequencySet(words_dict, sentence, minimal_support, minimal_size):
    words_in_sentence = len(sentence)
    result_groups = []

    # Составляем последовательности начиная с каждого слова
    for word_index in range(words_in_sentence):
        word_group = []
        for i in range(word_index, words_in_sentence):
            current_word = sentence[i]
            if(current_word == '"' or current_word == "'"):
                continue
            else:
                if(words_dict.get(current_word, 0) >= minimal_support):
                    word_group.append(current_word)
                else:
                    break

        if(len(word_group) >= minimal_size):
            result_groups.append(word_group)

    return result_groups


# Вычисляет поддерку групп слов в списке
def calculateGroupsSupport(group_list):
    result_dict = dict()

    for group in group_list:
        key = ''
        for word in group:
            key = key + ' ' + word
        key = key[1:]
        result_dict[key] = result_dict.get(key, 0) + 1
        
    return result_dict



def isSentencesContainsWord(sentences, test_word):
    for sentence in sentences:
        for word in sentence:
            if(str(word) == str(test_word)):
                return True
    return False

def count_of_words_in_sentences(sentences):
    counter = 0
    for sentence in sentences:
        for word in sentence:
            counter = counter + 1
    return counter

# Вычисляет IDF для каждого слова каждого текста и возвращает словарик СЛОВО:IDF
def calculateWordsIDF(texts, ):
    all_documents_count = len(texts);
    idf_data = dict()
    for text in texts:
        for word, frequency in text.word_frequency.items():
            word_doc_freq = 0.0;

            for doc in texts:
                if(isSentencesContainsWord(doc.register_pass_centences, word)):
                    word_doc_freq = word_doc_freq + 1.0
                    continue

            pre_idx = (0.0 + all_documents_count)/word_doc_freq
            inverse_document_frequency = math.log10(pre_idx)
            idf_data[word] = inverse_document_frequency
    return idf_data


# Вычисляет TF*IDF для каждого слова каждого текста и записывает в text.words_tf_idf[word]
def calculateTFIDF(texts):
    for text in texts:
        text.word_count = count_of_words_in_sentences(text.register_pass_centences)
        for word, frequency in text.word_frequency.items():
            tf = frequency/text.word_count
            text.words_tf_idf[word] = idf_word_data[word] * tf;

def writeWordTFIDFToString(idf_word_data):
    log_string = "Файлы\n"
    for text in texts:
        log_string = log_string + "\n" + text.filename + ";;;;"+'\n'
        log_string = log_string + 'Word; IDF; TF; IDF*TF;\n'

        for word, frequency in text.word_frequency.items():
            tf = frequency/text.word_count
            log_string = log_string + word + ";" + str(idf_word_data[word]) + ';' + str(tf) + ';' + str(text.words_tf_idf[word])  + ';\n'
    return log_string


def removeTFIDFWordsWithMiniamlMultiplier(texts , min_mult):
    for text in texts:
        sorted_TFIDF = sorted(text.words_tf_idf.items(), key=lambda x: x[1], reverse=True)  
        max_value = 0.0
        if(len(sorted_TFIDF)>0):
            max_value = sorted_TFIDF[0][1]

        minimal_value = min_mult*max_value
        for item in sorted_TFIDF:
            word = item[0]
            tfidf = item[1]
            
            if(tfidf < minimal_value):
                text.words_tf_idf.pop(word)
                text.word_frequency.pop(word)


### ПРОГРАММА  ---------------------------------------------------------------

# Получаем экземпляр анализатора (10-20мб)
morph = pymorphy2.MorphAnalyzer()

# Сюда будем записывать текст выходных файлов
log_string = ""


# Загружаем предложения из нескольких файлов
texts = []
for filename in input_filenames:
    text_data = TextData(filename)
    text_data.original_sentences = readSentencesFromInputText(filename)
    texts.append(text_data)


# Переводим предложения в списки слов (tokenized_sentence)
for text in texts:
    for sentence in text.original_sentences:
        if(len(sentence) > 0):
            tokenized_sentence = tokenizers.simple_word_tokenize(sentence)
            text.tokenized_sentences.append(tokenized_sentence)

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
calculateTFIDF(texts)





log_string = writeWordTFIDFToString(idf_word_data)
writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_6.csv')

# Вырезаем из TF-IDF 50% лучших слов
removeTFIDFWordsWithMiniamlMultiplier(texts , 0.25)

log_string = writeWordTFIDFToString(idf_word_data)
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
