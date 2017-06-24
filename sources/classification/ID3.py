# -*- coding: utf-8 -*-
import csv
import re, os
import numpy as np
import pymorphy2

##########################################
# directoryTest = 'china/test'
# category = list(os.listdir('china/train'))
#
# directoryTrue = 'china/train/' + category[0]
# directoryFalse = 'china/train/' + category[1]
#
# trueFiles = os.listdir(directoryTrue)
# falseFiles = os.listdir(directoryFalse)
# testFiles = os.listdir(directoryTest)
############################################
category = []
trueFiles = []
falseFiles=[]
testFiles = []
trueFilesName = []
falseFilesName=[]
testFilesName = []
input_dir1 = ''

def Classification_Text_ID3(input_dir,output_dir, trainingSet, trainingClass, testSet):

    global category
    global trueFiles
    global falseFiles
    global testFiles
    global trueFilesName
    global falseFilesName
    global testFilesName
    global input_dir1

    tree_file = output_dir + 'tree.txt'
    result_file = output_dir + 'result.txt'
    category = sorted(list(set(trainingClass)))
    input_dir1 = input_dir
    testFiles = testSet

    for i in range(len(trainingSet)):
       if  trainingClass[i] == category[0]:
           trueFiles.append(trainingSet[i])
       else:
           falseFiles.append(trainingSet[i])

    Words = words()
    list_words_in_file = Analysis_of_words(Words)
    applyID3(tree_file,Words,list_words_in_file)
    testTrue = []
    testFalse = []
    lastWord = ' '
    i = 0
    for f in testFiles:
        arr = list(set(f))
        with open(tree_file) as tree:
            for line in tree:
                s1 = line.strip()
                s1 = s1.split('=')
                if s1[0]==lastWord:
                    lastWord = ' '
                    continue
                if s1[0] in arr or lastWord != ' ' :
                    continue
                else:
                    if s1[0] == '0':
                        testFalse.append(testFilesName[i])
                        i+=1
                        break
                    if s1[0] == '1':
                        testTrue.append(testFilesName[i])
                        i+=1
                        break
                    else:
                        a = s1[0]
                        lastWord = a
                        continue
    res = open(result_file,'w')

    for i in range(len(testTrue)):
        if i == 0 :
            res.write('\n' + category[0] + ":")
            print('\n' + category[0] + ":")
        res.write(str('\n\t' + testTrue[i]))
        print(str('\t' + testTrue[i]))
    for i in range(len(testFalse)):
        if i == 0:
            res.write('\n' + category[1] + ":")
            print('\n' + category[1] + ":")
        res.write(str('\n\t' + testFalse[i]))
        print(str('\t' + testFalse[i]))
    res.close()
    category = []
    trueFiles = []
    falseFiles = []
    testFiles = []
    trueFilesName = []
    falseFilesName = []
    testFilesName = []
    input_dir1 = ''

def words():
    Words = {}
    global category
    global trueFiles
    global falseFiles
    global testFiles
    global trueFilesName
    global falseFilesName
    global testFilesName
    global input_dir1

    for f in trueFiles:
        arr = list(set(f))
        for s in arr:
            if s not in Words:  Words[s] = [0,1,0.0]
            else: Words[s][1]+=1

    for f in falseFiles:
        arr = f
        arr = list(set(arr))
        for s in arr:
            if s not in Words:  Words[s] = [1,0,0.0]
            else: Words[s][0]+=1
    return Words

def Analysis_of_words(Words):
    global category
    global trueFiles
    global falseFiles
    global testFiles
    global trueFilesName
    global falseFilesName
    global testFilesName
    global input_dir1

    list_words = sorted(Words.keys())
    list_words_in_files = {}
    cells = []

    trueFilesName = os.listdir(input_dir1 + '/train/' + category[0])
    falseFilesName = os.listdir(input_dir1 + '/train/' + category[1])
    testFilesName = sorted(os.listdir(input_dir1 + '/test/' + category[0]) +  os.listdir(input_dir1 + '/test/' + category[1]))

    for i in range(len(trueFiles)):
        for j in range(len(list_words) + 1): cells.append(0)

        arr = trueFiles[i]
        arr = list(set(arr))
        for s in list_words:
            if s in arr: cells[list_words.index(s)] = 1
        cells[len(cells)-1] = 1
        list_words_in_files[trueFilesName[i]] = cells
        cells = []

    for i in range(len(falseFiles)):
        for j in range(len(list_words) + 1): cells.append(0)
        arr = falseFiles[i]
        arr = list(set(arr))
        for s in list_words:
            if s in arr: cells[list_words.index(s)] = 1
        list_words_in_files[falseFilesName[i]] = cells
        cells = []

    return list_words_in_files

def ParseAttributes2(Words,list_words_in_file):
    global category
    global trueFiles
    global falseFiles
    global testFiles
    global trueFilesName
    global falseFilesName
    global testFilesName
    global input_dir1
    attr, attrnames, tests = {}, [], []
    justList = list_words_in_file
    attrnum = len(Words.keys()) + 1
    for s in Words.keys():
        i = 0
        fWords = [s, 1, 0]
        attrnames.append(fWords[0])
        attr[fWords[0]] = [i, fWords[1], fWords[2]]
        attr[fWords[1]] = 1
        attr[fWords[2]] = 0
        i+=1
    num = attrnum -1 # позиция вывода
    #testnum = len(trueFiles) + len(falseFiles)
    for s in justList.keys():
        tests.append(justList[s])
    return [attrnum, sorted(attrnames), attr, tests, num]

def entropy(tests,num):
    import math
    def log2(x): return math.log(x)/math.log(2)
    neg = float(len(list(filter(lambda x:(x[num]==0),tests))))
    tot = float(len(tests))
    if ((neg==tot) or (neg==0)): return 0
    # e = -(neg/tot)*log2(neg/tot)-((tot-neg)/tot)*log2(((tot-neg)/tot))
    return -(neg/tot)*log2(neg/tot)-((tot-neg)/tot)*log2(((tot-neg)/tot))

def gain(tests,attrnum,num):
    res = 0
    for i in range(2):
        arr = list(filter(lambda x:(x[attrnum]==i),tests))
        res += entropy(arr,num)*len(arr)/float(len(tests))
        # A = entropy(tests,num)-res
    return entropy(tests,num)-res

def ID3(tests,num,f,tabnum,usedattr,attrnames,attr):
    mas = ['', '', '', '', ]
    def findgains(x):
        if usedattr[x]: return 0
        return gain(tests,x,num)
    if (len(tests)==0):
        f.write('\t'*tabnum+'1')
        return
    if len(list(filter(lambda x:(x[num]==0),tests)))>len(list(filter(lambda x:(x[num]==1),tests))):
        majority = '0'
    else: majority = '1'
    gains = list(map(findgains, range(len(tests[0]))))
    maxgain = gains.index(max(gains))
    if (gains [maxgain] == 0):
        f.write('\t' * tabnum + majority + '\n')
        return
    arrpos = list(filter(lambda x: (x[maxgain] == 1), tests))
    arrneg = list(filter(lambda x: (x[maxgain] == 0), tests))
    newusedattr = usedattr
    newusedattr[maxgain] = True
    f.write('\t' * tabnum + str(attrnames[maxgain] + '=' + str(attr[attrnames[maxgain]][1]) + '\n'))

    if (len(arrpos) == 0):
        f.write('\t' * (tabnum + 1) + majority + '\n')
    else:
        ID3(arrpos, num, f, tabnum + 1, newusedattr, attrnames, attr)
    f.write('\t' * tabnum + str(attrnames[maxgain]) + '=' + str(attr[attrnames[maxgain]][2]) + '\n')
    if (len(arrneg) == 0):
        f.write('\t' * (tabnum + 1) + majority + '\n')
    else:
        ID3(arrneg, num, f, tabnum + 1, newusedattr, attrnames, attr)

def applyID3(outfname,Words,list_words_in_file):
    bigarr = ParseAttributes2(Words,list_words_in_file)
    attrnum, attrnames, attr, tests, num = bigarr[0], bigarr[1], bigarr[2], bigarr[3], bigarr[4]
    f = open(outfname, 'w')
    usedattr = []
    for i in range(attrnum): usedattr.append(i == num)
    ID3(tests, attrnum - 1, f, 0, usedattr, attrnames, attr)

########TEST############

# inputF = 'C:/Users/art-c/Desktop/TextStageProcessor-master/input_files/classification/china'
# outputF = 'C:/Users/art-c/Desktop/TextStageProcessor-master/output_files/classification/id3_out/'
# TESTSET = [['китайский', 'китайский', 'китайский', 'токио', 'япония']]
# CLASS = ['china', 'china', 'china', 'not_china']
# SET = [['китайский', 'пекин', 'китайский'], ['китайский', 'китайский', 'шанхай'], ['китайский', 'китайский', 'макао'], ['токио', 'япония', 'китайский']]

# Classification_Text_ID3(inputF,outputF,SET,CLASS,TESTSET )

