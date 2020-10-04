from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import v_measure_score
import numpy as np

from parameters import *
from knn_network import *

#A função purity recebe a matriz de confusão e a lista de todos os documentos e calcula a Pureza.
def purity(carray, list_glp):
    sum = np.sum(np.max(carray, axis=0))
    purity = sum / len(list_glp)
    return purity

#A função accuracy recebe a matriz de confusão e a lista de todos os documentos e calcula a Acurácia.
def accuracy(carray, list_glp):
    sum_tp = 0
    for i in range(0, len(carray)):
        sum_tp += carray[i][i]
    return sum_tp / len(list_glp)

#A função micro_precision recebe a matriz de confusão e a lista de todos os documentos e calcula a Micro-Precisão.
def micro_precision(carray, list_glp):
    sum_tp = 0
    for i in range(0, len(carray)):
        sum_tp += carray[i][i]
            
    micro_precision = sum_tp / len(list_glp)        
    return micro_precision

#A função macro_precision recebe a matriz de confusão e a lista de todos os documentos e calcula a Macro-Precisão.
def macro_precision(carray, list_glp):
    sum_tp = 0
    for i in range(0, len(carray)):
        total = 0
        for j in carray:
            total += j[i] 
        if(total == 0):
            continue
        sum_tp += carray[i][i] / total
        
    macro_precision = sum_tp / len(carray)
    return macro_precision

#A função micro_recall recebe a matriz de confusão e a lista de todos os documentos e calcula a Micro-Revocação.
def micro_recall(carray, list_glp):
    sum_tp = 0
    for i in range(0, len(carray)):
        sum_tp += carray[i][i]
    
    micro_recall = sum_tp / len(list_glp)    
    return micro_recall

#A função macro_recall recebe a matriz de confusão e a lista de todos os documentos e calcula a Macro-Revocação.
def macro_recall(carray, list_glp):
    sum_tp = 0
    for i in range(0, len(carray)):
        total = sum(carray[i])
        if(total == 0):
            continue
        sum_tp += carray[i][i] / total
    macro_recall = sum_tp / len(carray)
    return macro_recall

#A função micro_f1score recebe a precisão e revocação e calcula o Micro-Score F1.
def micro_f1score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

"""
A função macro_f1score recebe a matrix de confusão e uma lista única de todas as
classes existentes e calcula o Macro-Score F1.
"""
def macro_f1score(macro_precision, macro_recall):
    macro_f1 = micro_f1score(macro_precision, macro_recall)
    return macro_f1

#A função entropy recebe a lista de grupos e a lista de documentos e a lista de classes e calcula a Entropia  .
def entropy(glp, list_glp, lst_predict, classes):
    omega = 0
    #i = cluster.
    for i in glp:
	#j = documento de um cluster.
        plist = [j for j in i]
	#Lista de classes
        classes_list = [classes[int(j)] for j in plist]

        wc = []
        #[0..número de classes]
        for z in lst_predict:
            #Conta quantos elementos de cada classe e adiciona em wc.
            wc.append(classes_list.count(z))
        sum = 0
        #z = total de documentos de cada classe
        for z in wc:
            #número de documentos de uma classe / total
            if(z / len(list_glp) != 0):
                sum += (z / len(i)) * math.log2(z / len(i))
        omega += (-1 * sum) * np.sum(wc) / len(list_glp)
    return omega

"""
A função external_evaluate_metrics recebe os parâmetros:

carray: Matriz de confusão
glp: Lista de grupos encontrado pelo algoritmo de detecção de comunidade
list_glp: Lista de todos os documentos.

lst_correct: lista de classes corretas.
lst_predict: listas de classes previstas.

stdout: Diz se as medidas de avaliação externa serão mostradas na tela.
--------------------------------------------------------------------------------
E retorna todos as medidas de avaliação externas.
"""

def external_evaluate_metrics(carray, glp, list_glp, lst_correct, lst_predict, dict_classes):

    m_accuracy = accuracy(carray, list_glp)
    m_purity = purity(carray, list_glp)
    m_micro_precision = micro_precision(carray, list_glp)
    m_macro_precision = macro_precision(carray, list_glp)
    m_micro_recall = micro_recall(carray, list_glp)
    m_macro_recall = macro_recall(carray, list_glp)
    m_micro_f1score = micro_f1score(m_micro_precision, m_micro_recall)
    m_macro_f1score = macro_f1score(m_macro_precision, m_macro_recall)
    m_entropy = entropy(glp, list_glp, np.unique(lst_predict), dict_classes)

    m_adjusted_rand = adjusted_rand_score(lst_correct, lst_predict)
    m_completeness = completeness_score(lst_correct, lst_predict)
    m_fowlkes_mallows = fowlkes_mallows_score(lst_correct, lst_predict)
    m_homogeneity = homogeneity_score(lst_correct, lst_predict)
    m_mutual_info = mutual_info_score(lst_correct, lst_predict)
    m_v_measure = v_measure_score(lst_correct, lst_predict)
    
    if math.isnan(m_adjusted_rand):
        m_adjusted_rand = 0
    if math.isnan(m_completeness):
        m_completeness = 0
    if math.isnan(m_fowlkes_mallows):
        m_fowlkes_mallows = 0
    if math.isnan(m_mutual_info):
        m_mutual_info = 0
    if math.isnan(m_v_measure):
        m_v_measure = 0
    if math.isnan(m_macro_precision):
        m_macro_precision = 0
    if math.isnan(m_macro_recall):
        m_macro_recall = 0
    if math.isnan(m_macro_f1score):
        m_macro_f1score = 0

    return m_accuracy, m_purity, m_micro_precision, m_macro_precision, m_micro_recall, m_macro_recall, m_micro_f1score, m_macro_f1score, m_entropy, m_adjusted_rand, m_completeness, m_fowlkes_mallows, m_homogeneity, m_mutual_info, m_v_measure
