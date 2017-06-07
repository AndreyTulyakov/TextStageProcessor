#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import codecs
import math
import pymorphy2
from pymorphy2 import tokenizers 

from sources.TextData import TextData, readSentencesListFromInputText


def getCompiledFromSentencesText(sentences):
    result_string = ''
    for sentence in sentences:
        for word in sentence:
            result_string += (' ' + word)
    return result_string


# Читает текст из файла и возвращает список предложений (без запятых)
def readSentencesFromInputText(filename, input_dir_name):
    full_path = filename
    if(input_dir_name != None):
        full_path = input_dir_name + '/' + filename

    result = []
    sentences = readSentencesListFromInputText(full_path)
    for sentence in sentences:
        if(len(sentence)>0):
            result.append(sentence)
    return result


# Загружает из директории input_dir_name все txt файлы в список объектов TextData
def loadInputFilesFromList(input_files_list):
    texts = []
    for filename in input_files_list:
        text_data = TextData(filename)
        text_data.original_sentences = readSentencesFromInputText(filename)
        texts.append(text_data)

    return texts

# Загружает из директории input_dir_name все txt файлы в список объектов TextData
def loadInputFiles(input_dir_name):
    input_filenames = []
    for filename in os.listdir(input_dir_name):
        if filename.endswith(".txt"):
            input_filenames.append(filename)

    texts = []
    for filename in input_filenames:
        text_data = TextData(filename)
        text_data.original_sentences = readSentencesFromInputText(filename, input_dir_name)
        texts.append(text_data)

    return texts

def loadInputFilesFromList(filenames):
    texts = []
    for filename in filenames:
        text_data = TextData(filename[filename.rfind('/')+1:])
        text_data.full_filename = filename
        with codecs.open(filename, 'r', "utf-8") as text_file:
            data=text_file.read().replace('\n', ' ')
            sentences = data.split('.')
            for i in range(len(sentences)):
                sentences[i] = sentences[i].strip().replace(',', '')
            text_data.original_sentences = sentences

        texts.append(text_data)

    return texts


def tokenizeSingleText(text, configurations):
    minimal_words_in_sentence = 1
    if(configurations != None):
        minimal_words_in_sentence = configurations.get("minimal_words_in_sentence", 1)

    remove_index_list = []

    for index, sentence in enumerate(text.original_sentences):
        if (len(sentence) > 1):
            tokenized_sentence = tokenizers.simple_word_tokenize(sentence)
            updated_tokenized_sentence = []
            for word in tokenized_sentence:
                if word.isalpha() and len(word)>1:
                    updated_tokenized_sentence.append(word)
            tokenized_sentence = updated_tokenized_sentence

            if(len(tokenized_sentence) >= minimal_words_in_sentence):
                text.tokenized_sentences.append(tokenized_sentence)
            else:
                remove_index_list.append(index);
        else:
            remove_index_list.append(index);

    # Печать предложений перед вырезкой предложений
    # string_for_print = ''
    # for index, sentence in enumerate(text.original_sentences):
    #     string_for_print = string_for_print + str(index) + ')' + sentence + '\n'
    # writeStringToFile(string_for_print, 'output_files/preProcessing_before_cut.txt')
    #
    need_agresive_filtration = False
    if(configurations != None):
        need_agresive_filtration = configurations.get("need_agresive_filtration", False)

    sorted_remove_index_list = sorted(remove_index_list, key=lambda x: x, reverse=True)
    if(need_agresive_filtration):
        for index in sorted_remove_index_list:
            text.original_sentences.pop(index)

    # Печать предложений после вырезки
    # string_for_print = ''
    # for index, sentence in enumerate(text.original_sentences):
    #     string_for_print = string_for_print + str(index) + ')' + sentence + '\n'
    # writeStringToFile(string_for_print, 'output_files/preProcessing_after_cut.txt')

    return text.tokenized_sentences


def tokenizeTextData(texts, configurations = None):
    # Переводим предложения в списки слов (tokenized_sentence)
    for text in texts:
        text.tokenized_sentences = tokenizeSingleText(text, configurations)
    return texts


# Записывает/перезаписывает строку любой длины c переносами (data_str) в файл (filename)
def writeStringToFile(data_str, filename, strict_utf8_encoding=False):
    try:
        if(strict_utf8_encoding):
            with open(filename, 'w', encoding='utf-8') as out_text_file:
                out_text_file.write(data_str)
        else:
            with open(filename, 'w') as out_text_file:
                out_text_file.write(data_str)
    except PermissionError:
        print("ERROR!", "NO ACCESS TO FILE:", filename, ' - CLOSE OTHER APPLICATIONS')
        exit(-1)


# Определяет является ли слово частью ФИО (с вероятностью score)
def wordPersonDetector(word, morph):
    results = morph.parse(word)
    for result in results:
        if('Name' in result.tag
                or 'Surn' in result.tag
                or 'Patr' in result.tag):
            if(result.score >= 0.05):
                return True, results

    return False, results

def wordSurnameDetector(word, results):
    for result in results:
        if('Surn' in result.tag):
            if(result.score >= 0.05):
                return True
    return False

# Удаляет СТОП-СЛОВа (Предлоги, союзы и тд.)
def removeStopWordsFromSentences(sentences, morph, configurations):

    minimal_word_size = configurations.get("minimal_word_size", 3)
    cut_ADJ = configurations.get("cut_ADJ", False)
    for sentence in sentences:
        i = 0

        while i < len(sentence):
            current_word = sentence[i]

            if len(current_word) < minimal_word_size or current_word.isalpha() == False:
                 sentence.pop(i)
                 i = i - 1
            else:
                results = morph.parse(current_word)
                for result in results:
                    if((cut_ADJ and (result.tag.POS == 'ADJF'
                        or result.tag.POS == 'ADJS'))
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

    result_sentences = []
    for sentence in sentences:
        if(len(sentence) != 0):
            result_sentences.append(sentence)

    return result_sentences

def removeStopWordsInTexts(texts, morph, configurations):
    for text in texts:
        text.no_stop_words_sentences = removeStopWordsFromSentences(text.tokenized_sentences, morph, configurations)

    log_string = "Удаление стоп-слов:\n"

    for text in texts:
        log_string = log_string + '\nText:' + text.filename + '\n'
        for sentence in text.no_stop_words_sentences:
            for word in sentence:
                log_string = log_string + ' ' + word
            log_string = log_string + '\n'
    return texts, log_string

def normalizeTexts(texts, morph):
    log_string = "Нормализация:\n"
    for text in texts:
        log_string = log_string + '\nText:' + text.filename + '\n'
        for sentence in text.no_stop_words_sentences:
            current_sentence = []
            for word in sentence:
                isPerson, results = wordPersonDetector(word, morph)
                if(isPerson == False or (isPerson and wordSurnameDetector(word, results))):
                    result = results[0] # По умолчанию берем наиболее достоверный разбора слова
                    #current_sentence.append(word)
                    current_sentence.append(result.normal_form)
                    log_string = log_string + ' ' + result.normal_form
                else:

                    current_sentence.append(word)
                    log_string = log_string + ' ' + word
            log_string = log_string + '\n'
            text.normalized_sentences.append(current_sentence)
    return texts, log_string


def fixRegisterInTexts(texts, morph):
    log_string = "Приведение регистра:\n"
    for text in texts:
        log_string = log_string + '\nText:' + text.filename + '\n'
        for sentence in text.normalized_sentences:
            current_sentence = []
            for word in sentence:
                if(wordPersonDetector(word, morph) == True):
                    current_sentence.append(word.capitalize())
                else:
                    current_sentence.append(word.lower())
                log_string = log_string + ' ' + current_sentence[-1]
            text.register_pass_centences.append(current_sentence)
            log_string = log_string + '\n'
    return texts, log_string

def calculateWordsFrequencyInTexts(texts):

    for text in texts:
        for sentense in text.register_pass_centences:
            for word in sentense:
                    text.word_frequency[word] = text.word_frequency.get(word, 0) + 1

        # Сортируем слова по частоте
        text.sorted_word_frequency = sorted(text.word_frequency.items(), key=lambda x: x[1], reverse=True)

    log_string = 'Расчет частотной таблицы слов.\n'

    for text in texts:
        log_string = log_string + '\nText:' + text.filename + '\n'
        log_string = log_string + "Слово;Кол-во упоминаний\n"
        for key, value in text.sorted_word_frequency:
            log_string = log_string + key + ';' + str(value) + '\n'

    return texts, log_string


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
def calculateWordsIDF(texts):
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
def calculateTFIDF(texts, idf_word_data):
    for text in texts:
        text.word_count = count_of_words_in_sentences(text.register_pass_centences)
        for word, frequency in text.word_frequency.items():
            tf = frequency/text.word_count
            text.words_tf_idf[word] = idf_word_data[word] * tf;

def writeWordTFIDFToString(texts, idf_word_data):
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
            
            if tfidf < minimal_value:
                text.words_tf_idf.pop(word)
                text.word_frequency.pop(word)


def makePreprocessing(filenames, morph, configurations, additional_output=None):

    # Загружаем предложения из нескольких файлов
    texts = loadInputFilesFromList(filenames)
    output_dir = configurations.get("output_files_directory", "output_files") + "/preprocessing/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Разделяем предложения на слова
    texts = tokenizeTextData(texts)
    checkAdditionalOutput(additional_output,'Этап препроцессинга:\n')

    # Удаление стоп-слов из предложения (частицы, прилагательные и тд)
    checkAdditionalOutput(additional_output,'1) Удаление стоп-слов.\n')

    texts, log_string = removeStopWordsInTexts(texts, morph, configurations)
    writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_1.txt')

    # Переводим обычное предложение в нормализованное (каждое слово)
    checkAdditionalOutput(additional_output,'2) Нормализация.\n')
    texts, log_string = normalizeTexts(texts, morph)
    writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_2.txt')

    # Приведение регистра (все слова с маленькой буквы за исключением ФИО)
    checkAdditionalOutput(additional_output,'3) Приведение регистра.\n')
    texts, log_string = fixRegisterInTexts(texts, morph)
    writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_3.txt')

    # Подсчет частоты слов в тексте
    checkAdditionalOutput(additional_output,'4) Расчет частотной таблицы слов.\n')
    texts, log_string = calculateWordsFrequencyInTexts(texts)
    writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_4.csv')

    return texts

def checkAdditionalOutput(additional_output, text):
    if additional_output:
        additional_output.append(text)

# Для алгоритмов не требующих нормализации, удаления стоп-слов каких-либо модификаций слов
def makeFakePreprocessing(filenames):
    # Загружаем предложения из нескольких файлов
    texts = loadInputFilesFromList(filenames)
    texts = tokenizeTextData(texts)
    for text in texts:
        text.no_stop_words_sentences = text.tokenized_sentences
        text.normalized_sentences = text.tokenized_sentences
        text.register_pass_centences = text.tokenized_sentences
    texts, log_string = calculateWordsFrequencyInTexts(texts)
    return texts


def printMatrixToString(matrix, horizontal_header=None, vertical_header=None):
    result_string = ''
    rows, cols = matrix.shape

    if (horizontal_header != None):
        for i in range(len(horizontal_header)):
            result_string = result_string + str(horizontal_header[i]) + ',\t'
        result_string += '\n'

    for row in range(rows):
        if (vertical_header != None):
            result_string = result_string + str(vertical_header[row]) + '\t'

        for col in range(cols):
            result_string = result_string + str(matrix[row, col]) + ',\t'

        result_string += '\n'

    return result_string