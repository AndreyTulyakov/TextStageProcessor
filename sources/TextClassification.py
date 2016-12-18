#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import neighbors
from sklearn.linear_model import SGDClassifier
from sklearn import tree


from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic

def StrExtractNum(s):
    m = re.search('[0-9]+', s)
    return int(m.group(0))



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


    def makeClassification(self):
        self.textEdit.setText("")
     
        self.filenames.sort(key = StrExtractNum)      

        files_all = []
        for file in self.filenames:
            files_all.append(open(file, 'r').read())

        regex=re.compile(".*(_[a-z]).*")
        cl1 = [m.group(1) for l in self.filenames for m in [regex.search(l)] if m]
        targets_all = []
        for cl in cl1:
            targets_all.append(cl[1:])

        n = 17
        doc_train = files_all[:n]
        train_target = targets_all[:n]
        docs_test = files_all[n:]
        test_target = targets_all[n:]  




        text_clf = Pipeline( [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
        text_clf = text_clf.fit(doc_train, train_target)
        predicted_nb = text_clf.predict(docs_test)

        text_clf_knn = Pipeline( [('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', neighbors.KNeighborsClassifier())] )
        text_clf_knn = text_clf_knn.fit(doc_train, train_target)
        predicted_knn = text_clf_knn.predict(docs_test)



        text_clf_svm = Pipeline( [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier())] )
        #('clf', SGDClassifier(loss='hinge', penalty='l2',
        #                      alpha=1e-3, n_iter=5, random_state=42)),
        
        _ = text_clf_svm.fit(doc_train, train_target)
        predicted_svm = text_clf_svm.predict(docs_test)


        text_clf_tr = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', tree.DecisionTreeClassifier()),
        ])
        _ = text_clf_tr.fit(doc_train, train_target)
        predicted_tr = text_clf_tr.predict(docs_test)

        #print('Trained on files:')
        #print(self.filenames[:n])

        self.textEdit.append('Выборка для обучения:\n')
        for i in range(n):
            self.textEdit.append('   ' + self.filenames[i] + '\n')

        self.textEdit.append('Выборка для тестирования:\n')
        for i in range(n, len(self.filenames)):
            self.textEdit.append('   ' + self.filenames[i] + '\n')

        #print('Tested on files:') 
        #print(self.filenames[n:])
        self.textEdit.append('Алгоритм Naive Bayes:')
        self.textEdit.append(str(predicted_nb) + '\n')

        self.textEdit.append('Алгоритм K Nearest Neighbours:')
        self.textEdit.append(str(predicted_knn) + '\n')

        self.textEdit.append('Алгоритм Vector Machine:')
        self.textEdit.append(str(predicted_svm) + '\n')

        self.textEdit.append('Алгоритм Decision Trees:')
        self.textEdit.append(str(predicted_tr) + '\n')

        # print('Naive Bayes')
        # print(predicted_nb)
        # print('K Nearest Neighbours')
        # print(predicted_knn)
        # print('Support Vector Machine')
        # print(predicted_svm)
        # print('Decision Tree:')
        # print(predicted_tr)


        #from sklearn import metrics
        #print(metrics.classification_report(test_target, predicted,
        #    target_names=['n', 'x']))
        #print(metrics.confusion_matrix(test_target, predicted))
        #print(metrics.confusion_matrix(test_target, predicted_knn))
        #print(metrics.confusion_matrix(test_target, predicted_svm))
            

        self.textEdit.append('Успешно завершено.')

        QMessageBox.information(self, "Внимание", "Процесс классификации завершен!")
