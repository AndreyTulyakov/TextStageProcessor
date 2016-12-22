#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pymorphy2
import collections

from pymorphy2 import tokenizers 

from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic

from sources.TextData import TextData
from sources.TextPreprocessing import *


person_counter = 0
essence_counter = 0
action_counter = 0
deanon_stack = []
person_ids = dict()
essences = dict()
actions = dict()

def readSentencesFromInputText(filename):
    with open(filename, 'r') as text_file:

        # Читаем весь текст из файла.
        data=text_file.read().replace('\n', '')

        # Разделяем на предложения
        sentences = data.split('.')

        # Удаляем пробелы в начале и конце строки
        for i in range(len(sentences)):
            sentences[i] = sentences[i].strip().replace(',', '')

        return sentences
    return None


def en2ruPosName(enPosName):
    if(enPosName == None):
        return ''

    translated_words = dict({
        'NOUN':'Имя сушествительное',
        'ADJF':'Имя прилагательное',
        'ADJS':'Имя прилагательное (краткое)',
        'COMP':'Компаратив',
        'VERB':'Глагол (личная форма)',
        'INFN':'Глагол (инфинитив)',
        'PRTF':'Причастие (полное)',
        'PRTS':'Причастие (краткое)',
        'GRND':'Деепричастие',
        'NUMR':'Числительное',
        'ADVB':'Наречие',
        'NPRO':'Местоимение-существительное',
        'PRED':'Предикатив',
        'PREP':'Предлог',
        'CONJ':'Союз',
        'PRCL':'Частица',
        'INTJ':'Междометие'})
    return translated_words[enPosName]


def showAllWordPOS(morph, tokenized_sentences):
    i = 0
    for word in tsen:
        result = morph.parse(word)[0]
        if(result.tag.POS):
            print(str(i) + ') ' + word + ' (' + en2ruPosName(result.tag.POS) + ')')
        i = i + 1
    pass


"""
Проверяет что в предложении есть слова отвечающие за действия или события.
"""
def sentenceActionDetector(words_list, morph):
    for word in words_list:
        results = morph.parse(word)
        for result in results:
            if(result.tag.POS == 'VERB'
                    or result.tag.POS == 'INFN'
                    or result.tag.POS == 'PRTF'
                    or result.tag.POS == 'PRTS'
                    or result.tag.POS == 'GRND'):
                if(result.score >= 0.2):
                    return True
    return False



"""
Проверяет что в предложении есть местоимения или имена.
"""

def wordPersonDetector(word, morph):
    results = morph.parse(word)
    for result in results:
        if(result.tag.POS == 'NPRO'
                or 'Name' in result.tag
                or 'Surn' in result.tag
                or 'Patr' in result.tag):
            if(result.score >= 0.2):
                return True
    return False

def sentencePersonDetector(words_list, morph):
    for word in words_list:
        if(wordPersonDetector(word, morph)):
            return True;
    return False

# Проверяет является ли слово местоимением-существительным
def wordPersonNPRODetector(word, morph):
    results = morph.parse(word)
    for result in results:
        if(result.tag.POS == 'NPRO' and ('1per' not in morph.parse(word)[0].tag)):
            if(result.score >= 0.2):
                return True
    return False


def analyzePersons(sentence, morph):
    global person_ids

    # Деанонимизируем ОН, ОНА, ЕГО и тд
    print('1: ' + str(sentence))
    deanomized_sentence = deanonimizeSentence(sentence, deanon_stack, morph)
    #print('Deadnoned: ',deanomized_sentence)
    print('2: ' + str(deanomized_sentence))
    # Находим полные идентификаторы лиц, например: <Иван Васильевич>
    # Заменяем полные идентификаторы лиц кратким выражением #$1, #$2 и тд.
    personIdentificators, updated_sentence = extractFullPersonIdentificatorAndReplaceWithId(deanomized_sentence, morph)

    for key, value in personIdentificators.items():
        person_ids[key] = value

    #print(updated_sentence)

    normStrSen = ''
    normilized_sentence = []
    for word in updated_sentence:
        if word.startswith('#$'):
            alter_word = person_ids[word]
            normilized_sentence.append(alter_word)
            normStrSen = normStrSen + ' ' + str(alter_word)
        else:
            normilized_sentence.append(word)
            normStrSen = normStrSen + ' ' + word

    #print(normStrSen)    

    return updated_sentence    




def extractFullPersonIdentificatorAndReplaceWithId(sentence, morph):
    global deanon_stack
    global person_counter
    persons_identificators = collections.defaultdict(str)
    updated_sentence = []

    previous_word_is_person = False
    previous_word_case = ''
    currentIdentifier = []

    for word in sentence:

        if(wordPersonDetector(word, morph)):
            # Если встречается Мы, Я и тд

            if(previous_word_is_person == False):
                previous_word_is_person = True
                currentIdentifier.append(morph.parse(word)[0].inflect({'nomn'}).word)
                previous_word_case = morph.parse(word)[0].tag.case


            else:
                if(previous_word_case == morph.parse(word)[0].tag.case):
                    currentIdentifier.append(morph.parse(word)[0].inflect({'nomn'}).word)
                else:
                    person_id = str('#${0}').format(person_counter)

                    if(wordPersonNPRODetector(word, morph) == False):
                        deanon_stack.append([person_id, currentIdentifier])

                    persons_identificators[person_id] = currentIdentifier
                    
                    updated_sentence.append(person_id)
                    person_counter = person_counter + 1;
                    currentIdentifier = [morph.parse(word)[0].inflect({'nomn'}).word]
                    previous_word_case = morph.parse(word)[0].tag.case
        else:
            if(previous_word_is_person == True):
                person_id = str('#${0}').format(person_counter)
                persons_identificators[person_id] = currentIdentifier
                if(wordPersonNPRODetector(word, morph) == False):
                    deanon_stack.append([person_id, currentIdentifier])

                updated_sentence.append(person_id)
                person_counter = person_counter + 1;
                currentIdentifier = None
                previous_word_case = None
                previous_word_is_person = False

            updated_sentence.append(word)
            currentIdentifier = []

    if(previous_word_is_person == True):
        person_id = str('#${0}').format(person_counter)
        persons_identificators[person_id] = currentIdentifier
        person_counter = person_counter + 1;
        if(wordPersonNPRODetector(word, morph) == False):
            deanon_stack.append([person_id, currentIdentifier])

        updated_sentence.append(person_id)

    return persons_identificators, updated_sentence



def deanonimizeSentence(updated_sentence, deanon_stack, morph):

    result_sentence = []
 
    for word in updated_sentence:
        if(wordPersonNPRODetector(word, morph) and len(deanon_stack)>0):
            result_sentence.append(deanon_stack[-1][0])
        else:
            result_sentence.append(word)

    return result_sentence


def findSpecificPrefixEssences(sentence, morph):
    global essence_counter
    global essences
    updated_sentence = []

    current_essence = ''
    started_essence = False
    quote_opened = False
    previous_word = ''

    for word in sentence:
        if(word.startswith('#$') == True):
            updated_sentence.append(word)
            continue

        if(started_essence):

            if(word == '"'):
                if(quote_opened == False):
                    quote_opened = True
                    current_essence = current_essence + ' "'
                else:
                    quote_opened = False
                    current_essence = current_essence + '"'
                    essence_id = str('#ESS{0}').format(essence_counter)
                    essence_counter = essence_counter + 1
                    essences[essence_id] = current_essence
                    updated_sentence.append(essence_id)
                    started_essence = False
                    current_essence = ''
            else:
                if(current_essence[-1] != '"'): current_essence = current_essence + ' '
                current_essence = current_essence + word    
        else:
            prefix = word.upper()
            if(prefix == 'ОАО'
                or prefix == 'ЗАО'
                or prefix == 'ООО'):
                
                result = morph.parse(previous_word)[0]
                if(result.tag.POS == 'NOUN'):
                    current_essence = morph.parse(previous_word)[0].inflect({'nomn'}).word
                    updated_sentence.pop()

                current_essence = current_essence + ' ' + word
                started_essence = True


            else:
                current_essence = current_essence + ' ' + word
                updated_sentence.append(word)

        previous_word = word

    return updated_sentence


def findOtherEssences(sentence, morph):
    global essence_counter
    updated_sentence = []
    started_essence = False
    essence = ''

    for word in sentence:
        if(word.startswith('#$') or word.startswith('#ESS')):
                updated_sentence.append(word)
                continue

        result = morph.parse(word)[0]
        if(result.tag.POS == 'NOUN'):
            if(started_essence):
                essence = essence + ' ' + word
            else:
                essence = word
                started_essence = True
        else:
            if(started_essence):
                started_essence = False
                essence_id = str('#ESS{0}').format(essence_counter)
                essence_counter = essence_counter + 1
                essences[essence_id] = essence
                essence = ''
                updated_sentence.append(essence_id)
            else:
                updated_sentence.append(word)


    if(started_essence):
        started_essence = False
        essence_id = str('#ESS{0}').format(essence_counter)
        essence_counter = essence_counter + 1
        essences[essence_id] = essence
        essence = ''
        updated_sentence.append(essence_id)

    return updated_sentence



def findActions(sentence, morph):
    global action_counter
    updated_sentence = []

    for word in sentence:
        if(word.startswith('#$') or word.startswith('#ESS')):
                updated_sentence.append(word)
                continue

        result = morph.parse(word)[0]
        if(result.tag.POS == 'VERB'):
            action_id = str('#ACT{0}').format(action_counter)
            action_counter = action_counter + 1
            actions[action_id] = word
            updated_sentence.append(action_id)
        else:
            updated_sentence.append(word)
            pass
    return updated_sentence




# Decomposition and Rule Apply
class DialogConfigDRA(QDialog):

    def __init__(self, filenames, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogDecomposeAndRuleApply.ui', self)

        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.nu = []
        self.ns = []
        self.nv = []
        self.all_idf_word_keys = []
        self.texts = []

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonProcess.clicked.connect(self.process)
        self.textEdit.setText("")


    def process(self):
        self.textEdit.setText("")
        self.configurations["minimal_word_size"] = 1
        self.texts = makePreprocessing(self.filenames, self.morph, self.configurations, self.textEdit)
        output_dir = self.configurations.get("output_files_directory", "output_files")

        self.repaint()
        self.textEdit.append('Этап Анализа:\n')

        self.repaint()
        self.textEdit.append('\nПредложения после предварительной обработки:\n')
        for text in self.texts:
            updated_sentences = []
            for sentence in text.register_pass_centences:
                self.textEdit.append(str(sentence))
                current_sentence = analyzePersons(sentence, self.morph)
                updated_sentences.append(current_sentence)
            text.updated_sentences = updated_sentences


        self.textEdit.append('\nДействующие лица:\n')
        for item in person_ids.items():
            self.textEdit.append(str(item[0]) + ' -> ' + str(item[1]))


        for text in self.texts:
            updated_sentences_with_essenses = []
            for sentence in updated_sentences:
                current_sentence = findSpecificPrefixEssences(sentence, self.morph)
                current_sentence = findOtherEssences(current_sentence, self.morph)
                updated_sentences_with_essenses.append(current_sentence)
            text.updated_sentences = updated_sentences_with_essenses

        self.textEdit.append('\nИмеющиеся сущности:')
        for essence in essences.items():
            self.textEdit.append(str(essence[0]) + ' -> ' + str(essence[1]))

        for text in self.texts:
            final_primary_sentences = []
            for sentence in updated_sentences_with_essenses:
                current_sentence = findActions(sentence, self.morph)
                final_primary_sentences.append(current_sentence)
            text.updated_sentences = final_primary_sentences

        self.textEdit.append('\nНайденные события:')

        for action in actions.items():
            self.textEdit.append(str(action[0]) + ' -> ' + str(action[1]))

        self.textEdit.append('\nРезультирующий формат предложения:')
        for sentence in final_primary_sentences:
            self.textEdit.append(str(sentence) + '')





        self.textEdit.append('Успешно завершено.')
