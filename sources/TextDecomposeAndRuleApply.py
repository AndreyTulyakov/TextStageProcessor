#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2
import collections
import json

from PyQt5.QtCore import Qt
from pymorphy2 import tokenizers

from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic

from sources.TextData import TextData
from sources.TextPreprocessing import *
from sources.utils import Profiler

person_counter = 0
essence_counter = 0
action_counter = 0
deanon_stack = []
person_ids = dict()
essences = dict()
actions = dict()


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
    deanomized_sentence = deanonimizeSentence(sentence, deanon_stack, morph)

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
        if(result.tag.POS == 'VERB' or result.tag.POS == 'INFN'):
            action_id = str('#ACT{0}').format(action_counter)
            action_counter = action_counter + 1
            actions[action_id] = word
            updated_sentence.append(action_id)
        else:
            updated_sentence.append(word)
            pass
    return updated_sentence

"""
 Необходим разбор предложения:
 Переносим события в вид функции
 Покинул(а, с, d)
 Заменил(a, b)

 Заносим в стек преобразователя и на выходе получаем:
 Покинул(а, с, d)
 Вступил(b, c, d)

 Восстанавливаем таблицу событий.
"""
def formActions(sentences, actions_grammatic):

    functions_stack = []

    for sentence in sentences:
        current_function = []
        modified_sentence = sentence[:]

        print(sentence)

        # Поиск функции начинается с поиска идентификаторов действий
        for word in sentence:
            if(word.startswith('#ACT')):
                current_function.append(word)


        # Если идентификаторы функции имеются то удаляем его из предложения
        if(len(current_function)>0):
            modified_sentence.remove(current_function[0])
        else:
            continue

        current_gramatic = None
        if(len(current_function)>0):
            for action_grammatic in actions_grammatic:
                #print('LA:', action_grammatic[0].lower(), actions[current_function[0]].lower())
                if(action_grammatic[0].lower() == actions[current_function[0]].lower()):
                    current_gramatic = action_grammatic[:]
                    break;

        if(current_gramatic == None):
            print('Функция:', current_function[0], '[', actions[current_function[0]] ,'] не найдена!')
            break

        current_function[0] = actions[current_function[0]].lower()

        current_gramatic.pop(0)
        arg_counter = 0
        for argument in current_gramatic:

            if(argument == '#$'):
                for word in modified_sentence:
                    if(word.startswith('#$')):
                        current_function.append(word)
                        modified_sentence.remove(word)
                        arg_counter = arg_counter + 1
                        break;

            if(argument == '#ESS'):
                for word in modified_sentence:
                    if(word.startswith('#ESS')):
                        current_function.append(word)
                        modified_sentence.remove(word)
                        arg_counter = arg_counter + 1
                        break;

        if(arg_counter + 1 != len(current_function)):
            print('Ошибка разбора #701')
            break;
        else:
            if(len(current_function)> 0):
                functions_stack.append(current_function)

    return functions_stack




def applyRules(output_rules, actions_map):

    for output_rule in output_rules:
        input_functions = output_rule[0]
        func_index = 0
        while func_index < (len(actions_map)-len(input_functions)+1):
            rule_activated = True
            for input_func_index in range(len(input_functions)):
                if(actions_map[func_index+input_func_index][0] != input_functions[input_func_index][0]):
                    rule_activated = False
                    #print('Плохо', actions_map[func_index+input_func_index][0], input_functions[input_func_index][0])
                    break;

            if(rule_activated == True):
                local_func_list = actions_map[func_index:len(input_functions)+func_index]
                #print('Хорошо', output_rule, ' ---+---',local_func_list)
                result = applyRule(output_rule, local_func_list)
                #print('Хорошо2', output_rule)
                #print('result:', result)
                actions_map = actions_map[:func_index] + result + actions_map[func_index+len(input_functions):]
                func_index = 0
            else:
                func_index = func_index + 1

    return actions_map





def applyRule(output_rule, target_actions):
    func_counter = 0
    arg_counter = 0
    data = dict()

    for input_function in output_rule[0]:
        arg_counter = 0
        
        for arg in input_function:
            if(arg_counter >= len(target_actions[func_counter])):
                continue
            data[arg] = target_actions[func_counter][arg_counter]
            arg_counter = arg_counter + 1

        func_counter = func_counter + 1

    output_functions = []
    for output_function_in_rule in output_rule[1]:

        current_function = [output_function_in_rule[0]]
        
        for word in output_function_in_rule[1:]:
            result = data.get(word)
            if(result == None):
                continue
            current_function.append(result)

        output_functions.append(current_function)

    return output_functions


def reconstructFunctions(functions_with_aliases):

    result_func_list = []

    for row in functions_with_aliases:
        result_func = [row[0]]
        for i in range(1, len(row)):
            current_word = row[i]

            if(current_word.startswith('#$')):
                person_full_name = person_ids[current_word]
                person_str_name = ''
                for name in person_full_name:
                    person_str_name = person_str_name + " " + name.capitalize()
                person_str_name = person_str_name[1:]
                result_func.append(person_str_name)
                continue

            if(current_word.startswith('#ESS')):
                result_func.append(essences[current_word])
                continue

        result_func_list.append(result_func)

    return result_func_list


# Decomposition and Rule Apply
class DialogConfigDRA(QDialog):

    def __init__(self, filenames, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogDecomposeAndRuleApply.ui', self)

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint;
        self.setWindowFlags(flags)

        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.profiler = Profiler()

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

        self.profiler.start()
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

        for text in self.texts:
            self.textEdit.append('\nРезультирующий формат предложения:')
            for sentence in text.updated_sentences:
                self.textEdit.append(str(sentence) + '')


        self.textEdit.append('\nПреобразование в функциональный вид:')



        # правила формирования аргументов для каждой функции
        actions_grammatic = []
        self.rules_filename = self.filenames[0][:self.filenames[0].rfind('/')] + '/grammatic.json'
        with open(self.rules_filename) as rules_file:
            actions_grammatic = json.load(rules_file)

        for text in self.texts:
            actions_map = formActions(text.updated_sentences, actions_grammatic)
            for action in actions_map:
                self.textEdit.append('\nФункция:' + str(action))

        self.textEdit.append('\nПрименение правил вывода:')
        


        # Добавим правило вывода: [входные функции] [выходные функции]
        output_rules = []
        self.rules_filename = self.filenames[0][:self.filenames[0].rfind('/')] + '/rules.json'
        with open(self.rules_filename) as rules_file:
            output_rules = json.load(rules_file)

        output_rules_p = []

        for rule in output_rules:
            output_rules_p.append([rule['source_scheme'], rule['replace_scheme']])

        output_rules = output_rules_p
        updated_functions = applyRules(output_rules, actions_map)
        for function in updated_functions:
            self.textEdit.append('\n'+str(function))

        self.textEdit.append('\nВосстановление ссылок:')
        reconstructed_functions = reconstructFunctions(updated_functions)
        for function in reconstructed_functions:
            self.textEdit.append(str(function))


        self.textEdit.append('Успешно завершено.')
        self.textEdit.append('Выполнено за ' + self.profiler.stop() + ' с.')
