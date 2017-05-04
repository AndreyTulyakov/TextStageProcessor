#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Объединить все предложения в один список
from itertools import combinations

from pandas import DataFrame
import pandas

from sources.TextPreprocessing import writeStringToFile
from sources.apriori import Apriori

def join_all_texts_to_sentences_list(texts):
    sentenses = []
    for text in texts:
        for sentense in text.register_pass_centences:
            sentenses.append(sentense)
    return sentenses

# Отсортировать слова в каждом предложении
def sort_all_words_in_sentences(sentences):
    sorted_sentences = []
    for sentence in sentences:
        sorted_sentence = sorted(sentence, key=str.lower)
        sorted_sentences.append(sorted_sentence)
    return sorted_sentences

# Пихаем в dataframe
def save_2d_list_to_dataframe(data):
    df = DataFrame(data)
    return df


# Apriori
def apriori_alg(trans, support=0.01, minlen=2):
    print('appr_1')
    dna = trans.unstack().dropna()
    print('appr_2')
    ts = pandas.get_dummies(dna).groupby(level=1).sum()
    print('appr_3')
    collen, rowlen = ts.shape
    pattern = []
    for cnum in range(minlen, rowlen + 1):
        for cols in combinations(ts, cnum):
            print('cnum', cnum)
            patsup = ts[list(cols)].all(axis=1).sum()
            patsup = float(patsup) / collen
            pattern.append([",".join(cols), patsup])
    print('appr_4')
    sdf = pandas.DataFrame(pattern, columns=["Pattern", "Support"])
    print('appr_5')
    results = sdf[sdf.Support >= support]
    print('appr_6')
    return results

# Выполняем Apriori от и до
def makeAprioriForTexts(texts, output_dir, minsup = 0.01, minconf = 0.01):
    sentences = join_all_texts_to_sentences_list(texts)
    sentences = sort_all_words_in_sentences(sentences)

    ap = Apriori(sentences, minsup, minconf)
    ap.run()

    frequent_itemset = ap.save_to_csv_string_frequent_itemset()
    frequent_itemset = frequent_itemset.replace('.', ',')
    writeStringToFile(frequent_itemset, output_dir + 'apriori_frequent_itemset.csv')

    rules_string = ap.save_to_csv_string_rules()
    rules_string = rules_string.replace('.', ',')
    writeStringToFile(rules_string, output_dir + 'apriori_rules.csv')
