#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2
import math
from pymorphy2 import tokenizers
import os
<<<<<<< HEAD

from TextData import TextData

from TextPreprocessing import *


=======

# Считывает имена всех .txt файлов из входной директории
input_dir_name = "input_files"
input_filenames = []
for file in os.listdir(input_dir_name):
    if file.endswith(".txt"):
        input_filenames.append(input_dir_name + '/' + file);


### ФУНКЦИИ ------------------------------------------------------------------
>>>>>>> 9c0e7e938f947fdd05511279e660759bc980bc8d

# Dmitry code

def sim(D, i, j):
    summ = 0
    summ_sqrt_i = 0
    summ_sqrt_j = 0
    for k in range(len(D[0])):
        summ += D[i][k] * D[j][k]

    for k in range(len(D[0])):
        summ_sqrt_i += D[i][k] * D[i][k]
    summ_sqrt_i = math.sqrt(summ_sqrt_i)

    for k in range(len(D[0])):
        summ_sqrt_j += D[j][k] * D[j][k]
    summ_sqrt_j = math.sqrt(summ_sqrt_j)

    sim = summ / (summ_sqrt_i * summ_sqrt_j)
    return sim


def GetS(D):
    n = len(D)
    m = len(D[0])
    S = [[0] * m for i in range(n)]

    for row in range(n):
        for item in range(m):
            if (item < row):
                S[row][item] = sim(D, row, item)
            else:
                S[row][item] = 0

    return S


def agglomerative_hierarchical_clustering(D):
    S = GetS(D)
    F = []
    for k in range(len(D)):
        F[k] = 1
    A = []
    for k in range(len(D)):
        F[k] = 1

    return 0


<<<<<<< HEAD
=======
# Читает текст из файла и возвращает список предложений (без запятых)
def readSentencesFromInputText(filename):
    with open(filename, 'r') as text_file:
        data = text_file.read().replace('\n', ' ')
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
        if ('Name' in result.tag
            or 'Surn' in result.tag
            or 'Patr' in result.tag):
            if (result.score >= 0.05):
                return True
    return False


# Удаляет СТОП-СЛОВа (Предлоги, союзы и тд.)
def removeStopWordsFromSentences(sentences, morph):
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            current_word = sentence[i]
            results = morph.parse(current_word)
            for result in results:
                if (result.tag.POS == 'ADJF'
                    or result.tag.POS == 'ADJS'
                    or result.tag.POS == 'PREP'
                    or result.tag.POS == 'ADVB'
                    or result.tag.POS == 'COMP'
                    or result.tag.POS == 'CONJ'
                    or result.tag.POS == 'PRCL'):
                    if (result.score >= 0.25):
                        sentence.pop(i)
                        i = i - 1
                        break
            i = i + 1
    return sentences


# Проверяет является ли слово местоимением-существительным (Он, Она и тд.)
def wordPersonNPRODetector(word, morph):
    results = morph.parse(word)
    for result in results:
        if (result.tag.POS == 'NPRO'):
            if (result.score >= 0.2):
                return True
    return False


# Заменяет ОН, ОНА и тд. на последнюю упомянутую персону в предложении.
def deanonimizeSentence(updated_sentence, deanon_stack, morph):
    result_sentence = []
    for word in updated_sentence:
        if (wordPersonNPRODetector(word, morph) and len(deanon_stack) > 0):
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
            if (current_word == '"' or current_word == "'"):
                continue
            else:
                if (words_dict.get(current_word, 0) >= minimal_support):
                    word_group.append(current_word)
                else:
                    break

        if (len(word_group) >= minimal_size):
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


>>>>>>> 9c0e7e938f947fdd05511279e660759bc980bc8d
def CombineAll():
    # Объединить все тексты
    for filename in input_filenames:
        with open(filename, 'r') as text_file:
            data = text_file.read()
            with open('combined.txt', 'a') as out_text_file:
                out_text_file.write(data)
    return 0


def dist(w1, w2):
    k = len(w1)
    dist = [0 for x in range(k)]
    summ = 0
    for k in range(len(t_all)):
        summ += math.pow(w1[k] - w2[k], 2)
    dist[k] = math.sqrt(summ)
    return dist


<<<<<<< HEAD
=======
### ПРОГРАММА  ---------------------------------------------------------------

# CombineAll()
# Получаем экземпляр анализатора (10-20мб)
morph = pymorphy2.MorphAnalyzer()

# Сюда будем записывать текст выходных файлов
log_string = ""


>>>>>>> 9c0e7e938f947fdd05511279e660759bc980bc8d
def FindUnion(Dist, F):
    min_i = 0
    min_j = 0
    min = 1000
    n = len(Dist)
    m = len(Dist[0])
    for i in range(n):
        for j in range(i):
            if ((Dist[i][j] < min) & (F[i] == 1) & (j < i)):
                min = Dist[i][j]
                min_i = i
                min_j = j

    return [min_i, min_j]


<<<<<<< HEAD
=======
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


>>>>>>> 9c0e7e938f947fdd05511279e660759bc980bc8d
class Cluster:
    ap = 1 / 2
    aq = 1 / 2
    b = 0
    g = 0

    def __init__(self, doc, dist):
        self.docs = [{doc}]
        self.Dist = dist

    def AddDoc(self, doc, dist, n):
        i = 0
        self.docs.append(doc)
        newDist = []
        for k in range(len(dist)):
            newDist[k] = Cluster.ap * (self.Dist[k]) + Cluster.aq * (dist[k]) + Cluster.b * (
                self.Dist[n]) + Cluster.g * (math.fabs(self.Dist[k] - dist[k]))


<<<<<<< HEAD
def printDist(dist, texts, filenames):
    dist_string = ''

    for name in filenames:
        dist_string = dist_string + '; doc' + str(name)[str(name).find('/') + 1:str(name).find('.')]
    dist_string += '\n'
    for i in range(len(texts)):
        dist_string += 'doc' + str(filenames[i])[
                               str(filenames[i]).find('/') + 1:str(filenames[i]).find('.')]
=======
# Загружаем предложения из нескольких файлов
texts = []
for filename in input_filenames:
    text_data = TextData(filename)
    text_data.original_sentences = readSentencesFromInputText(filename)
    texts.append(text_data)

# Переводим предложения в списки слов (tokenized_sentence)
for text in texts:
    for sentence in text.original_sentences:
        if (len(sentence) > 0):
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
            if (wordPersonDetector(word, morph) == False):
                result = morph.parse(word)[0]  # По умолчанию берем наиболее достоверный разбора слова
                # current_sentence.append(word)
                current_sentence.append(result.normal_form)
                log_string = log_string + ' ' + result.normal_form
            else:
                current_sentence.append(word)
                log_string = log_string + ' ' + word
        log_string = log_string + '\n'
        text.normalized_sentences.append(current_sentence)
    print(text.filename + ' completed')

writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_1.txt')

# Удаление стоп-слов из предложения (частицы, прилагательные и тд)
for text in texts:
    text.no_stop_words_sentences = removeStopWordsFromSentences(text.normalized_sentences, morph)
    # text.no_stop_words_sentences = text.normalized_sentences

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
            if (wordPersonDetector(word, morph) == True):
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

    text.sorted_word_frequency = sorted(text.word_frequency.items(), key=lambda x: x[1], reverse=True)
#
#
# print('4) Расчет частотной таблицы слов.')
# log_string = 'Расчет частотной таблицы слов.\n'
#
# for text in texts:
#     log_string = log_string + '\nText:' + text.filename + '\n'
#     log_string = log_string + "Слово;Кол-во упоминаний\n"
#     for key, value in text.sorted_word_frequency:
#         log_string = log_string + key + ';' + str(value) + '\n'
# writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_4.csv')

# Нахождение матрицы весов

t_all = dict()

for text in texts:
    for key, value in text.sorted_word_frequency:
        t_all[key] = t_all.get(key, 0) + 1

# Найти df
df_string = ''
df_string = df_string + "Слово;Используется в документах\n"
for key, value in t_all.items():
    df_string = df_string + key + ';' + str(value) + '\n'
writeStringToFile(df_string.replace('\n ', '\n'), 'df.csv')

W = [[0 for x in range(len(t_all))] for y in range(len(texts))]
print('len(texts)=' + str(len(texts)))
print('len(t_all)=' + str(len(t_all)))
W_norm = [0 for x in range(len(texts))]
i = 0
j = 0
for row in range(len(texts)):
    j = 0
    for key, value in t_all.items():
        text = texts[row]
        if (key in text.word_frequency):
            frequency_in_this_doc = text.word_frequency[key]
        else:
            frequency_in_this_doc = 0
        W[i][j] = frequency_in_this_doc * math.log10(len(texts) / value)
        W_norm[i] += math.pow(W[i][j], 2)
        print('W[' + key + '][' + input_filenames[i] + '] = ' + str(frequency_in_this_doc) + '*Log(' + str(
            len(texts)) + '/' + str(value) + ') = ' + str(W[i][j]))

        j += 1
    W_norm[i] = math.sqrt(W_norm[i])
    print('wnorm = ' + str(W_norm[i]))
    i += 1

for i in range(len(texts)):
    for j in range(len(t_all)):
        W[i][j] /= W_norm[i]

W_string = ''
W_string = W_string + "Нормированные веса\n"
for key, value in t_all.items():
    W_string = W_string + ';' + key
W_string += '\n'
i = 0
for row in W:
    W_string += input_filenames[i]
    for item in row:
        W_string = W_string + ';' + str(round(item, 10))
    W_string += '\n'
    i += 1
writeStringToFile(W_string.replace('\n ', '\n'), 'W.csv')

S = GetS(W)
sim_string = ''
for name in input_filenames:
    sim_string = sim_string + ';' + name
sim_string += '\n'
for i in range(len(texts)):
    sim_string += input_filenames[i]
    for j in range(len(t_all)):
        sim_string = sim_string + ';' + str(S[i][j])
    sim_string += '\n'
writeStringToFile(sim_string.replace('\n ', '\n'), 'sim.csv')

n = len(texts)
m = len(texts)
dist = [[0 for x in range(n)] for y in range(m)]

for i in range(n):
    for j in range(m):
        summ = 0
        for k in range(len(t_all)):
            summ += math.pow(W[i][k] - W[j][k], 2)
        dist[i][j] = math.sqrt(summ)


def printDist(dist):
    dist_string = ''

    for name in input_filenames:
        dist_string = dist_string + '; doc' + str(name)[str(name).find('/') + 1:str(name).find('.')]
    dist_string += '\n'
    for i in range(len(texts)):
        dist_string += 'doc' + str(input_filenames[i])[
                               str(input_filenames[i]).find('/') + 1:str(input_filenames[i]).find('.')]
>>>>>>> 9c0e7e938f947fdd05511279e660759bc980bc8d
        for j in range(len(texts)):
            dist_string = dist_string + '; ' + str(round(dist[i][j], 2))
        dist_string += '\n'
    print(dist_string)


<<<<<<< HEAD
=======
dist_string = ''

for name in input_filenames:
    dist_string = dist_string + ';doc' + str(name)[str(name).find('/') + 1:str(name).find('.')]
dist_string += '\n'
for i in range(len(texts)):
    dist_string += 'doc' + str(input_filenames[i])[
                           str(input_filenames[i]).find('/') + 1:str(input_filenames[i]).find('.')]
    for j in range(len(texts)):
        dist_string = dist_string + ';' + str(round(dist[i][j], 2))
    dist_string += '\n'
writeStringToFile(dist_string.replace('\n ', '\n').replace('.', ','), 'dist.csv')

doc2cluster = [0 for x in range(len(texts))]
for i in range(len(texts)):
    doc2cluster[i] = i

clusters = dict()
for i in range(len(texts)):
    clusters[i] = [i]


>>>>>>> 9c0e7e938f947fdd05511279e660759bc980bc8d
def ClusterByDoc(x, clusters):
    for key, value in clusters.items():
        for i in range(len(clusters[key])):
            if (clusters[key][i] == x):
                return key


def UnionClusters(cluster1, cluster2, clusters):
    for i in range(len(clusters[cluster2])):
        clusters[cluster1].append(clusters[cluster2][i])
    clusters.pop(cluster2)


def Cluster2String(clusters, cluster):
    res = '{'
    for i in range(len(clusters[cluster])):
        res += str(clusters[cluster][i]+1)
        if(i<len(clusters[cluster])-1):
            res+=','
    res += '}'
    return res


<<<<<<< HEAD

def makeClasterization(filenames, morph, configurations):

    texts = makePreprocessing(filenames, morph, configurations)
    output_dir = configurations.get("output_files_directory", "output_files") + "/clasterization/" 

    # Нахождение матрицы весов

    t_all = dict()

    for text in texts:
        for key, value in text.sorted_word_frequency:
            t_all[key] = t_all.get(key, 0) + 1

    # Найти df
    df_string = ''
    df_string = df_string + "Слово;Используется в документах\n"
    for key, value in t_all.items():
        df_string = df_string + key + ';' + str(value) + '\n'
    writeStringToFile(df_string.replace('\n ', '\n'), output_dir + 'df.csv')

    W = [[0 for x in range(len(t_all))] for y in range(len(texts))]
    print('len(texts)=' + str(len(texts)))
    print('len(t_all)=' + str(len(t_all)))
    W_norm = [0 for x in range(len(texts))]
    i = 0
    j = 0
    for row in range(len(texts)):
        j = 0
        for key, value in t_all.items():
            text = texts[row]
            if (key in text.word_frequency):
                frequency_in_this_doc = text.word_frequency[key]
            else:
                frequency_in_this_doc = 0
            W[i][j] = frequency_in_this_doc * math.log10(len(texts) / value)
            W_norm[i] += math.pow(W[i][j], 2)
            #print('W[' + key + '][' + filenames[i] + '] = ' + str(frequency_in_this_doc) + '*Log(' + str(
            #    len(texts)) + '/' + str(value) + ') = ' + str(W[i][j]))

            j += 1
        W_norm[i] = math.sqrt(W_norm[i])
        print('wnorm = ' + str(W_norm[i]))
        i += 1

    for i in range(len(texts)):
        for j in range(len(t_all)):
            W[i][j] /= W_norm[i]

    W_string = ''
    W_string = W_string + "Нормированные веса\n"
    for key, value in t_all.items():
        W_string = W_string + ';' + key
    W_string += '\n'
    i = 0
    for row in W:
        W_string += filenames[i]
        for item in row:
            W_string = W_string + ';' + str(round(item, 10))
        W_string += '\n'
        i += 1
    writeStringToFile(W_string.replace('\n ', '\n'), output_dir + 'W.csv')

    S = GetS(W)
    sim_string = ''
    for name in filenames:
        sim_string = sim_string + ';' + name
    sim_string += '\n'
    for i in range(len(texts)):
        sim_string += filenames[i]
        for j in range(len(t_all)):
            sim_string = sim_string + ';' + str(S[i][j])
        sim_string += '\n'
    writeStringToFile(sim_string.replace('\n ', '\n'), output_dir + 'sim.csv')

    n = len(texts)
    m = len(texts)
    dist = [[0 for x in range(n)] for y in range(m)]

    for i in range(n):
        for j in range(m):
            summ = 0
            for k in range(len(t_all)):
                summ += math.pow(W[i][k] - W[j][k], 2)
            dist[i][j] = math.sqrt(summ)





    dist_string = ''

    for name in filenames:
        dist_string = dist_string + ';doc' + str(name)[str(name).find('/') + 1:str(name).find('.')]
    dist_string += '\n'
    for i in range(len(texts)):
        dist_string += 'doc' + str(filenames[i])[
                               str(filenames[i]).find('/') + 1:str(filenames[i]).find('.')]
        for j in range(len(texts)):
            dist_string = dist_string + ';' + str(round(dist[i][j], 2))
        dist_string += '\n'
    writeStringToFile(dist_string.replace('\n ', '\n').replace('.', ','), output_dir + 'dist.csv')

    doc2cluster = [0 for x in range(len(texts))]
    for i in range(len(texts)):
        doc2cluster[i] = i

    clusters = dict()
    for i in range(len(texts)):
        clusters[i] = [i]




    F = [1 for x in range(len(texts))]

    result = ''

    for k in range(len(texts) - 1):
        printDist(dist, texts, filenames)
        union = FindUnion(dist, F)
        firstCluster = ClusterByDoc(union[0], clusters)
        secondCluster = ClusterByDoc(union[1], clusters)
        print('found clusters = ' + str(firstCluster) + ' and ' + str(secondCluster))
        # print(str(union[0] + 1) + ' + ' + str(union[1] + 1) + ' with dist= ' + str(dist[union[0]][union[1]]))
        result += 'step' + str(k) + ';' + Cluster2String(clusters, firstCluster) + ';+;' \
                  + Cluster2String(clusters, secondCluster) + ';=;'
        UnionClusters(firstCluster, secondCluster, clusters)
        result += Cluster2String(clusters, ClusterByDoc(union[0], clusters)) + '\n'
        # doc2cluster[union[1]] = '{' + doc2cluster[union[1]] + ',' + doc2cluster[union[0]] + '}'
        # doc2cluster[union[0]] = ''

        F[union[0]] = 0
        # F[union[1]] = 0
        new_dist = [[0 for x in range(len(texts))] for y in range(len(texts))]
        for i in range(len(texts)):
            for j in range(len(texts)):
                new_dist[i][j] = dist[i][j]

        for j in range(len(texts)):
            for i in range(2):
                new_dist[j][union[i]] = 0.5 * (dist[j][union[0]]) + 0.5 * (dist[j][union[1]])

        dist = new_dist

    writeStringToFile(result.replace('\n ', '\n'), output_dir + 'steps.csv')

    for i in range(len(texts)):
        print(doc2cluster[i])
=======
F = [1 for x in range(len(texts))]

result = ''

for k in range(len(texts) - 1):
    printDist(dist)
    union = FindUnion(dist, F)
    firstCluster = ClusterByDoc(union[0], clusters)
    secondCluster = ClusterByDoc(union[1], clusters)
    print('found clusters = ' + str(firstCluster) + ' and ' + str(secondCluster))
    # print(str(union[0] + 1) + ' + ' + str(union[1] + 1) + ' with dist= ' + str(dist[union[0]][union[1]]))
    result += 'step' + str(k) + ';' + Cluster2String(clusters, firstCluster) + ';+;' \
              + Cluster2String(clusters, secondCluster) + ';=;'
    UnionClusters(firstCluster, secondCluster, clusters)
    result += Cluster2String(clusters, ClusterByDoc(union[0], clusters)) + '\n'
    # doc2cluster[union[1]] = '{' + doc2cluster[union[1]] + ',' + doc2cluster[union[0]] + '}'
    # doc2cluster[union[0]] = ''

    F[union[0]] = 0
    # F[union[1]] = 0
    new_dist = [[0 for x in range(len(texts))] for y in range(len(texts))]
    for i in range(len(texts)):
        for j in range(len(texts)):
            new_dist[i][j] = dist[i][j]

    for j in range(len(texts)):
        for i in range(2):
            new_dist[j][union[i]] = 0.5 * (dist[j][union[0]]) + 0.5 * (dist[j][union[1]])

    dist = new_dist

writeStringToFile(result.replace('\n ', '\n'), 'steps.csv')

for i in range(len(texts)):
    print(doc2cluster[i])


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

def isSentencesContainsWord(sentences, test_word):
    for sentence in sentences:
        for word in sentence:
            if (str(word) == str(test_word)):
                return True
    return False


def count_of_words_in_sentences(sentences):
    counter = 0
    for sentence in sentences:
        for word in sentence:
            counter = counter + 1
    return counter

# print('6) Вычисление модели TF*IDF.')
#
# all_documents_count = len(texts);
# idf_word_data = dict()
#
# for text in texts:
#
#     for word, frequency in text.word_frequency.items():
#         word_doc_freq = 0.0;
#
#         for doc in texts:
#             if (isSentencesContainsWord(doc.register_pass_centences, word)):
#                 word_doc_freq = word_doc_freq + 1.0
#                 continue
#
#         pre_idx = (0.0 + all_documents_count) / word_doc_freq
#         inverse_document_frequency = math.log10(pre_idx)
#         idf_word_data[word] = inverse_document_frequency
#
# sorted_IDF = sorted(idf_word_data.items(), key=lambda x: x[1], reverse=False)
#
# # for word, idf in sorted_IDF:
# #    print(word + " IDF:" + str(idf))
#
# for text in texts:
#     text.word_count = count_of_words_in_sentences(text.register_pass_centences)
#
# log_string = "Файлы\n"
#
# for text in texts:
#
#     log_string = log_string + "\n" + text.filename + ";;;;" + '\n'
#     log_string = log_string + 'Word; IDF; TF; IDF*TF;\n'
#
#     # print('DOC:' + text.filename)
#
#     for word, frequency in text.word_frequency.items():
#         tf = frequency / text.word_count
#         idf_tf = idf_word_data[word] * tf;
#         # print(word + " IDF_TF:" + str(idf_tf))
#         log_string = log_string + word + ";" + str(idf_word_data[word]) + ';' + str(tf) + ';' + str(idf_tf) + ';\n'
#
# writeStringToFile(log_string.replace('\n ', '\n'), 'output_stage_6.csv')


# Пока выходим
>>>>>>> 9c0e7e938f947fdd05511279e660759bc980bc8d
