# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import pymorphy2

idx_lbl = 'idx'
morph = pymorphy2.MorphAnalyzer() 

class Rocchio:
    
    def __init__(self, class_labels, tdict):
        
        self.k = len(class_labels)
        self.centroids = []
        self.lbl_dict = dict(zip(class_labels, range(self.k)))
        self.class_labels = class_labels
        self.tdict = tdict
        self.ctermcnt = np.zeros((self.k, 1))

    #находим центроиды классов
    def train(self, token_pool, tfidf_but_smoothing = True):

        #количество слов в классе
        for term, data in self.tdict.items():
            for cl in self.lbl_dict:
                if cl in data:
                    self.ctermcnt[self.lbl_dict[cl], 0] += data[cl]

        for cl in self.class_labels:
            self.centroids.append(np.zeros((len(self.tdict), 1)))
            for doc in token_pool[cl]:
                vec = self.__createNormalizedVectorRepresentation(doc, cl)
                self.centroids[self.lbl_dict[cl]] += vec

            self.centroids[self.lbl_dict[cl]] /= len(token_pool[cl])

    def centrouds(self):
        return(self.centroids)
        
    def lbl_dict(self):
        return(self.lbl_dict)
            

    #результатом будет документ с минимальным расстоянием
    def predict(self, doc):

        doc_vec = self.__createNormalizedVectorRepresentation(doc, None)

        distances = []
        for i in range(self.k):
            distances.append(np.linalg.norm(doc_vec - self.centroids[i]))

        return self.class_labels[distances.index(min(distances))]


    #тест на всех документах
    def predictPool(self, doc_collection):

        lbl_pool = {}
        for cl in self.class_labels:
            lbl_pool[cl] = []
            for doc in doc_collection[cl]:
                lbl_pool[cl].append(self.predict(doc))

        return lbl_pool

    #функция рассчёта усреднённого  вектора
    def __createNormalizedVectorRepresentation(self, tokens_list, cl = None, tfidf = True):

        vec = np.zeros((len(self.tdict), 1))
        for token in tokens_list:
            if token in self.tdict:
                vec[self.tdict[token][idx_lbl], 0] += 1

        token_set = set(tokens_list)
        if tfidf:
            if cl != None:
                for term in token_set:
                    if cl in self.tdict[term]:
                        vec[self.tdict[term][idx_lbl], 0] *= np.log(self.ctermcnt[self.lbl_dict[cl], 0] * 1.0 / self.tdict[term][cl])


        norm_vec = np.linalg.norm(vec)
        vec = (vec / (norm_vec + 1e-14))
        return vec


#%%

#возвращаем список документов и их слов
def createTokenPool(classes, paths):
    token_pool = {}
    for cl in classes:
        token_pool[cl] = []
        for path in paths[cl]:
            token_pool[cl].append(tokenizeDoc(path))

    return token_pool
    
#форматирование документа
def tokenizeDoc(doc_address, min_len = 0, remove_numerics=True):
    tokens = []
    try:
        f = open(doc_address, "r", encoding='utf-8')
        text = f.read().lower()
        text = ''.join(e for e in text if e.isalpha() or e.isspace())
        words = re.split("\s", text)
        words = f_tokenizer(words)
        r = re.compile("[^a-zA-z]+")
        tokens = [w for w in filter(r.match, words)]
        f.close()
    except:
        print("Error: %s couldn't be opened!", doc_address)
    finally:
        return tokens
#нормализация документа
def f_tokenizer(s):
    f = []
    for j in s:
        m = morph.parse(j.replace('.',''))
        if len(m) != 0:
            wrd = m[0]
            if wrd.tag.POS not in ('NUMR','PREP','CONJ','PRCL','INTJ'):
                f.append(wrd.normal_form)
    return f
    
#создание словаря классов
def createDictionary(classes, tokens_pool):
    token_dict = {}
    idx = 0 
    for cl in classes:
        for tokens_list in tokens_pool[cl]:
            for token in tokens_list:
                if token in token_dict:             
                    if cl in token_dict[token]:
                        token_dict[token][cl] += 1
                    else:
                        token_dict[token][cl] = 1
                else:
                    token_dict[token] = {}
                    token_dict[token][idx_lbl] = idx
                    idx += 1
                    token_dict[token][cl] = 1
    return token_dict



