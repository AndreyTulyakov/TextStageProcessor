# -*- coding: utf-8 -*-
import csv
import re, os
import numpy as np
#import csv
#import operator
import pymorphy2

def localize_floats(row):
    return [
        str(el).replace('.', ',') if isinstance(el, float) else el 
        for el in row
    ]

def dictOfDictToCsv(dct, fpath):
    with open(fpath, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter = ';', lineterminator = '\n')
        for key, value in dct.items():
            writer.writerow([key])
            for key2, value2 in value.items():
                writer.writerow([key2, str(value2).replace('.',',')])
                
def dictToCsv(dct, fpath):
    with open(fpath, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=';', lineterminator='\n')
        for key, value in dct.items():
            writer.writerow([key, str(value).replace('.',',')])
            
def dictListToCsv(dct, fpath):
    with open(fpath, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter = ';', lineterminator = '\n')
        for key, value in dct.items():
           writer.writerow([key])
           scores = [x[1] for x in value]
           writer.writerow(["Принадлежит классу:" + value[np.argmax(scores)][0]])
           for el in value:
                writer.writerow(localize_floats(el))


def count_words(words):
    wc = {}
    for word in words:
        wc[word] = wc.get(word, 0.0) + 1.0
    return wc


#класс для хранения наборов слов
class BagOfWords(object):

    
    def __init__(self):
        self.__number_of_words = 0
        self.__bag_of_words = {}
    
    #сложение двух классов BagOfWords
    def __add__(self,other):

        erg = BagOfWords()
        sum = erg.__bag_of_words
        for key in self.__bag_of_words:
            sum[key] = self.__bag_of_words[key]
            if key in other.__bag_of_words:
                sum[key] += other.__bag_of_words[key]
        for key in other.__bag_of_words:
            if key not in sum:
                sum[key] = other.__bag_of_words[key]
        return erg
        
    #добавление слова
    def add_word(self,word):

        self.__number_of_words += 1
        if word in self.__bag_of_words:
            self.__bag_of_words[word] += 1
        else:
            self.__bag_of_words[word] = 1
    
    def len(self):

        return len(self.__bag_of_words)
    
    def Words(self):

        return self.__bag_of_words.keys()
    
        
    def BagOfWords(self):

        return self.__bag_of_words
       
    #частота конкретного слова
    def WordFreq(self,word):

        if word in self.__bag_of_words:
            return self.__bag_of_words[word]
        else:
            return 0
            
#класс для хранения документов
class Document(object):

    _vocabulary = BagOfWords()
 
    def __init__(self, vocabulary):
        self.__name = ""
        self.__document_class = None
        self._words_and_freq = BagOfWords()
        Document._vocabulary = vocabulary
    
    #чтение документа
    def read_document(self, filename, morph, learn=False):
        try:
            text = open(filename,"r", encoding='utf-8').read()
        except UnicodeDecodeError:
            text = open(filename,"r", encoding='latin-1').read()
        text = text.lower()
        text = ''.join(e for e in text if e.isalpha() or e.isspace())
        words = re.split("\s", text)
        words = self.f_tokenizer(words, morph)
        r = re.compile("[^a-zA-z]+")
        words = [w for w in filter(r.match, words)]

        self._number_of_words = 0
        for word in words:
            self._words_and_freq.add_word(word)
            if learn:
                Document._vocabulary.add_word(word)
            
    #форматирование документа
    def f_tokenizer(self, s, morph):
        f = []
        for j in s:
            m = morph.parse(j.replace('.',''))
            if len(m) != 0:
                wrd = m[0]
                if wrd.tag.POS not in ('NUMR','PREP','CONJ','PRCL','INTJ'):
                    f.append(wrd.normal_form)
        return f
    
    #сложение двух классов Document
    def __add__(self,other):

        res = Document(Document._vocabulary)
        res._words_and_freq = self._words_and_freq + other._words_and_freq    
        return res
    
    def vocabulary_length(self):

        return len(Document._vocabulary)
                
    def WordsAndFreq(self):

        return self._words_and_freq.BagOfWords()
        
    def Words(self):

        d =  self._words_and_freq.BagOfWords()
        return d.keys()
    
    def WordFreq(self,word):

        bow =  self._words_and_freq.BagOfWords()
        if word in bow:
            return bow[word]
        else:
            return 0
        
    #пересечение двух классов Document
    def __and__(self, other):
  
        intersection = []
        words1 = self.Words()
        for word in other.Words():
            if word in words1:
                intersection += [word]
        return intersection
        
#класс для хранения всех документов одного класса
class DocumentClass(Document):
    def __init__(self, vocabulary):
        Document.__init__(self, vocabulary)
        self._number_of_docs = 0

    def __add__(self, other):
        res = DocumentClass(self._vocabulary)
        res._words_and_freq = self._words_and_freq + other._words_and_freq 
        return res

    def SetNumberOfDocs(self, number):
        self._number_of_docs = number
    
    def NumberOfDocuments(self):
        return self._number_of_docs

#класс для классификации
class Pool(object):
    
    def __init__(self):
        self.__document_classes = {}
        self.__vocabulary = BagOfWords()
        self.__morph = pymorphy2.MorphAnalyzer() 
        
    def BagOfWords_in_class(self, dclass):
        return self.__document_classes[dclass].WordsAndFreq()

    #количество слов в классе
    def sum_words_in_class(self, dclass):

        sum = 0
        for word in self.__vocabulary.Words():
            WaF = self.__document_classes[dclass].WordsAndFreq()
            if word in WaF:
                sum +=  WaF[word]
        return sum
    
    #обучение
    def learn(self, directory, dclass_name):

        x = DocumentClass(self.__vocabulary)
        dir = os.listdir(directory)
        for file in dir:
            d = Document(self.__vocabulary)
            d.read_document(directory + "/" +  file, self.__morph, learn = True)
            x = x + d
        self.__document_classes[dclass_name] = x
        x.SetNumberOfDocs(len(dir))

    #тест - перебор всех комбинаций классов
    def Probability(self, doc, dclass = ""):

        #если нет класса, вызываем Probability ещё раз и считаем его
        if dclass:
            sum_dclass = self.sum_words_in_class(dclass)
            prob = 0
        
            d = Document(self.__vocabulary)
            d.read_document(doc, self.__morph)
            
            #вероятность встретить документ d среди всех документов класса j;
            for j in self.__document_classes:
                sum_j = self.sum_words_in_class(j)
                prod = 1
                #произведение условных вероятностей всех слов входящих в d
                for i in d.Words():
                    wf_dclass = 1 + self.__document_classes[dclass].WordFreq(i)
                    wf = 1 + self.__document_classes[j].WordFreq(i)
                    r = wf * sum_dclass / (wf_dclass * sum_j)
                    prod *= r
                #Формула наивного байеса
                prob += prod * self.__document_classes[j].NumberOfDocuments() / self.__document_classes[dclass].NumberOfDocuments()
            if prob != 0:
                return round(1 / prob, 3)
            else:
                return -1
        else:
            prob_list = []
            #для каждого класса вызываем Probability
            for dclass in self.__document_classes:
                prob = self.Probability(doc, dclass)
                prob_list.append([dclass,prob])
            prob_list.sort(key = lambda x: x[1], reverse = True)
            return prob_list

    #процент похожих слов одного документа со всеми классами
    def DocumentIntersectionWithClasses(self, doc_name):
        res = [doc_name]
        for dc in self.__document_classes:
            d = Document(self.__vocabulary)
            d.read_document(doc_name, self.__morph, learn=False)
            o = self.__document_classes[dc] &  d
            intersection_ratio = len(o) / len(d.Words())
            res += (dc, intersection_ratio)
        return res
