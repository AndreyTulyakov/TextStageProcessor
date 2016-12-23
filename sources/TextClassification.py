#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from sources.TextPreprocessing import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import neighbors
from sklearn.linear_model import SGDClassifier
from sklearn import tree


from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic



class BagOfWords(object):

    
    def __init__(self):
        self.__number_of_words = 0
        self.__bag_of_words = {}
        
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
        
    def WordFreq(self,word):

        if word in self.__bag_of_words:
            return self.__bag_of_words[word]
        else:
            return 0
            
class Document(object):

    _vocabulary = BagOfWords()
 
    def __init__(self, vocabulary):
        self.__name = ""
        self.__document_class = None
        self._words_and_freq = BagOfWords()
        Document._vocabulary = vocabulary
    
    def read_document(self,filename, learn=False):

        try:
            text = open(filename,"r", encoding='utf-8').read()
        except UnicodeDecodeError:
            text = open(filename,"r", encoding='latin-1').read()
        text = text.lower()
        words = re.split("[^\wäöüÄÖÜß]*",text)

        self._number_of_words = 0
        for word in words:
            self._words_and_freq.add_word(word)
            if learn:
                Document._vocabulary.add_word(word)

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
                
    def __and__(self, other):
  
        intersection = []
        words1 = self.Words()
        for word in other.Words():
            if word in words1:
                intersection += [word]
        return intersection
        
class DocumentClass(Document):
    def __init__(self, vocabulary):
        Document.__init__(self, vocabulary)
        self._number_of_docs = 0

    def Probability(self,word):

        voc_len = Document._vocabulary.len()
        SumN = 0
        for i in range(voc_len):
            SumN = DocumentClass._vocabulary.WordFreq(word)
        N = self._words_and_freq.WordFreq(word)
        erg = 1 + N
        erg /= voc_len + SumN
        return erg

    def __add__(self,other):

        res = DocumentClass(self._vocabulary)
        res._words_and_freq = self._words_and_freq + other._words_and_freq 
 
        return res

    def SetNumberOfDocs(self, number):
        self._number_of_docs = number
    
    def NumberOfDocuments(self):
        return self._number_of_docs

class Pool(object):
    def __init__(self):
        self.__document_classes = {}
        self.__vocabulary = BagOfWords()
            
    def sum_words_in_class(self, dclass):

        sum = 0
        for word in self.__vocabulary.Words():
            WaF = self.__document_classes[dclass].WordsAndFreq()
            if word in WaF:
                sum +=  WaF[word]
        return sum
    
    def learn(self, directory, dclass_name):

        x = DocumentClass(self.__vocabulary)
        dir = os.listdir(directory)
        for file in dir:
            d = Document(self.__vocabulary)
            print(directory + "/" + file)
            d.read_document(directory + "/" +  file, learn = True)
            x = x + d
        self.__document_classes[dclass_name] = x
        x.SetNumberOfDocs(len(dir))

    
    def Probability(self, doc, dclass = ""):

        if dclass:
            sum_dclass = self.sum_words_in_class(dclass)
            prob = 0
        
            d = Document(self.__vocabulary)
            d.read_document(doc)

            for j in self.__document_classes:
                sum_j = self.sum_words_in_class(j)
                prod = 1
                for i in d.Words():
                    wf_dclass = 1 + self.__document_classes[dclass].WordFreq(i)
                    wf = 1 + self.__document_classes[j].WordFreq(i)
                    r = wf * sum_dclass / (wf_dclass * sum_j)
                    prod *= r
                prob += prod * self.__document_classes[j].NumberOfDocuments() / self.__document_classes[dclass].NumberOfDocuments()
            if prob != 0:
                return round(1 / prob, 3)
            else:
                return -1
        else:
            prob_list = []
            for dclass in self.__document_classes:
                prob = self.Probability(doc, dclass)
                prob_list.append([dclass,prob])
            prob_list.sort(key = lambda x: x[1], reverse = True)
            return prob_list

    def DocumentIntersectionWithClasses(self, doc_name):
        res = [doc_name]
        for dc in self.__document_classes:
            d = Document(self.__vocabulary)
            d.read_document(doc_name, learn=False)
            o = self.__document_classes[dc] &  d
            intersection_ratio = len(o) / len(d.Words())
            res += (dc, intersection_ratio)
        return res



class DialogConfigClassification(QDialog):

    def __init__(self, filenames, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogConfigClassification.ui', self)

        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonClassify.clicked.connect(self.makeClassification)

        self.textEdit.setText("")


    def writeStringToFile(data_str, filename):
        with open(filename, 'w') as out_text_file:
            out_text_file.write(data_str)

    def makeClassification(self):
        self.textEdit.setText("")
        base = "input_files/classification/test/"

        DClasses = os.listdir(base)

        p = Pool()
        for i in DClasses:
            p.learn(base + i, i)

        log_string = "Naive Bayes:\n"
        for i in DClasses:
            dir = os.listdir(base + i)
            for file in dir:
                res = p.Probability(base + i + "/" + file)
                str_out = i + ": " + file + ": " + str(res)
                self.textEdit.setText(str_out)
                log_string = log_string + str_out + '\n'

        self.textEdit.setText(log_string)
        

                
        output_dir = self.configurations.get("output_files_directory", "output_files/")

        writeStringToFile(log_string, output_dir + 'output_naive_bayes.txt')


        self.textEdit.append('Успешно завершено.')

        QMessageBox.information(self, "Внимание", "Процесс классификации завершен!")
