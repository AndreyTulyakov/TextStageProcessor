# -*- coding: utf-8 -*-
import os
import re
import math
import pymorphy2
import numpy as np
import operator
from collections import defaultdict
import csv

morph = pymorphy2.MorphAnalyzer() 

#чтение путей файлов и классов
def makeFileList(root_path = 'input_files/classification/', fread = True, fprocess = True):
    type_data = []
    split = 0
    files = []
    train_folder = "train/"
    if len(root_path)>0 and root_path[-1] != '/':
        train_folder = '/' + train_folder
    path_train = root_path + train_folder
    folders = [path_train + folder + '/' for folder in os.listdir(path_train)]
    class_titles = os.listdir(path_train)

    for folder, title in zip(folders, class_titles):
        new_files = [folder + f for f in os.listdir(folder)]
        files.append(new_files)
        type_data.append([title] * len(new_files))
        split += len(new_files)

    test_folder = "test/"
    if len(root_path) > 0 and root_path[-1] != '/':
        test_folder = '/' + test_folder
    path_test = root_path + test_folder

    for cl in class_titles:
         if not os.path.exists(path_test + cl):
            os.makedirs(path_test + cl)

    folders = [path_test + folder + '/' for folder in os.listdir(path_test)]
    files_tst = []
    for folder, title in zip(folders, class_titles):
        new_files = [folder + f for f in os.listdir(folder)]
        files_tst.append(new_files)
        type_data.append([title] * len(new_files))

    type_data = sum(type_data, [])

    out = sum(files + files_tst, [])
    if(fread):
        out = createTokenPool(out, fprocess)

    return(out, type_data, split)



#чтение путей файлов и классов
def makeFileListLib(root_path = 'input_files/classification/', fread = True, fprocess = True):
    type_data = []
    split = 0
    files = []
    train_folder = "train/"
    if len(root_path)>0 and root_path[-1] != '/':
        train_folder = '/' + train_folder
    path_train = root_path + train_folder
    folders = [path_train + folder + '/' for folder in os.listdir(path_train)]
    class_titles = os.listdir(path_train)

    for folder, title in zip(folders, class_titles):
        new_files = [folder + f for f in os.listdir(folder)]
        files.append(new_files)
        type_data.append([title] * len(new_files))
        split += len(new_files)

    test_folder = "test/"
    if len(root_path) > 0 and root_path[-1] != '/':
        test_folder = '/' + test_folder
    path_test = root_path + test_folder

    for cl in class_titles:
         if not os.path.exists(path_test + cl):
            os.makedirs(path_test + cl)

    folders = [path_test + folder + '/' for folder in os.listdir(path_test)]
    files_tst = []
    for folder, title in zip(folders, class_titles):
        new_files = [folder + f for f in os.listdir(folder)]
        files_tst.append(new_files)
        type_data.append([title] * len(new_files))

    type_data = sum(type_data, [])

    out = sum(files + files_tst, [])
    filenames = out
    if(fread):
        out = createTokenPoolLib(out, fprocess)

    return(out, type_data, split, filenames)


#разделяем все документы на слова
def createTokenPool(paths, fprocess):
    token_pool = []
    for path in paths:
        token_pool.append(tokenizeDoc(path, fprocess))
    return token_pool

def createTokenPoolLib(paths, fprocess):
    token_pool = []
    for path in paths:
        try:
            f = open(path, "r", encoding='utf-8')
            text = f.read().lower()
            token_pool.append(text)
            f.close()
        except Exception as err:
            print("Ошибка чтения!", path, err)

    return token_pool

#разделяем документ на слова   
def tokenizeDoc(doc_address, fprocess = True):
    tokens = []
    try:
        f = open(doc_address, "r", encoding='utf-8')
        text = f.read().lower()
        text = ''.join(e for e in text if e.isalpha() or e.isspace())
        words = re.split("\s", text)
        if fprocess:
            words = f_tokenizer(words)
        words = list(filter(None, words))
        r = re.compile("[^a-zA-z]+")
        tokens = [w for w in filter(r.match, words)]
        f.close()
    except Exception as err:
        print("Ошибка чтения!", doc_address, err)
    finally:
        return tokens

#если нужно, проводим нормализацию
def f_tokenizer(s):
    f = []
    for j in s:
        m = morph.parse(j.replace('.',''))
        if len(m) != 0:
            wrd = m[0]
            if wrd.tag.POS not in ('NUMR','PREP','CONJ','PRCL','INTJ'):
                f.append(wrd.normal_form)
    return f

# Записывает/перезаписывает строку любой длины c переносами (data_str) в файл (filename)
def writeStringToFile2(data_str, filename):
    with open(filename, 'w') as out_text_file:
        out_text_file.write(data_str)
    print(filename)


# длина вектора
def normVec(nparray):
    return(np.sqrt(np.sum(np.power(nparray, 2))))


#матрица tf-idf
def makeTFIDF(data_train, data_test = None):
    words_in_doc = []
    for doc in data_train:
        words_in_doc.append(list(set(doc)))
    
    all_words = sum(words_in_doc, [])
    uniq_words = list(set(all_words))
    
    df_doc = []
    for word in uniq_words:
        df_doc.append(all_words.count(word))
    df_doc = np.array(df_doc)
    
    
    if data_test:
        data_tf = data_train + data_test
    tf = []
    for doc in data_tf:
        tfi = []
        for word in uniq_words:
            tfi.append(doc.count(word))
        tf.append(tfi)
    
    
    D = len(data_train)
    
    tfidf = []
    for tfi in tf:
        w = np.array(tfi) * np.log10(D / df_doc)
        row = w / normVec(w)
        tfidf.append(row.round(2).tolist())
    
    return(tfidf, uniq_words)
    
def addClassToTFIDF(matrix, vector):
    for i in range(len(matrix)):
        matrix[i].append(vector[i])
    return(matrix)

 #расстояние между векторами
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
 
#возвращает всех соседей и расстояния до них
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors, distances
 
 #процент ошибок
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def getBasePath(pathList):
    for i in range(len(pathList)):
        pathList[i] = os.path.basename(pathList[i])
    return pathList