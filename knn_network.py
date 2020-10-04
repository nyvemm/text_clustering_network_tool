import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import random

from random import randint
from base64 import b16encode
from sklearn.metrics import confusion_matrix

from parameters import *

"""
Esta função recebe os valores da similiaridade do cosseno e gera a rede KNN
com base nos k-vizinhos próximos. Caso não for necessário gerar a rede completa
o parâmetro 'limit' limita a quantidade de nós gerados
"""

#A função nindex, remove o número da classe de uma String.
def nindex(str):
    #O loop percorre até encontrar um dígito então retorna a sub-string de [0..n]
    for i in str:
        if i.isdigit():
            return str[0:str.index(i)]
    return None

"""
A função find_cluster_class recebe uma lista contendo um documento pertencente
a grupo  encontrada depois do algoritmo de agrupamento(glp) e encontra qual é a 
classe do sub-grupo contando qual foi a classe mais prevista.
"""
def find_cluster_class(lst):
    #Dicionário que contém o número de documentos de cada classe. 
    dict_cluster = {}
    #Para cada elemento na lista.
    for i in lst:
        #Se o documento não está no dicionário.
        if i not in dict_cluster.keys():
            dict_cluster[i] = 0
        else:
            dict_cluster[i] = dict_cluster[i] + 1
    #Retorna a classe com maior número de documentos
    return int(max(dict_cluster))

#Define os parâmetros da função antes de chamá-la.
ebc_normalized = True
ebc_weight = None
def most_central_edge(G):
    centrality = nx.edge_betweenness_centrality(G, weight = ebc_weight, normalized = ebc_normalized)
    max_cent = max(centrality.values())
    
    #Escala os valores de centralidade entre 0 e 1 e adiciona um ruído aleatório.
    centrality = {e: c / max_cent for e, c in centrality.items()}
    centrality = {e: c + random.random() for e, c in centrality.items()}
    return max(centrality, key = centrality.get)


"""
A função knn_network recebe as varíaveis:

list_cossine: lista contendo os valores com todos os valores 
de similaridade com seus vizinhos

dict_classes: dicionário que contém para cada documento n, a sua respectiva
classe, sendo n o seu indíce.

limit: representa o limite de iterações

--------------------------------------------------------------------------------
Cria e retorna um grafo utilizando os valores de similaridade de cosseno, relacionando
os seus k-vizinhos mais próximos (mais similares).
"""

def knn_network(list_cossine, dict_classes, k, weighted = False, limit = -1):
    G = nx.Graph()    
    #Verifica se há um limite de iterações.
    if limit == -1:
        limit = len(list_cossine)

    for index, key in enumerate(list_cossine):
        #Cria a lista de arestas.
        edge_list = []
        max_k = k
        #Ajusta o número de vizinhos para o máximo de documentos caso, o mesmo ultrapasse o valor.
        if(len(key) < k):
            max_k = len(key)
        #Cria a aresta entre cada documento e seus k-vizinhos
        for i in range(max_k):
            if weighted:
                edge_list.append((str(index), str(key[i][0]), key[i][1]))
            else:
                edge_list.append((str(index), str(key[i][0])))
                
        #Adiciona as arestas a lista de arestas
        if(weighted):
            G.add_weighted_edges_from(edge_list)
        else:
            G.add_edges_from(edge_list)
        #Se a iteração chegou ao limite, ela finalizará.
        if index == limit:
            break
        
    return G

#Gera um valor hexadecimal a partir de uma lista de cores.
def generate_and_convert_color(list):
    r = randint(0,255)
    b = randint(0,255)
    g = randint(0,255)    
    hex = '#%02x%02x%02x' % (r,g,b)
    
    #Se o valor hex já existir na lista de hex então o processo se repetirá.
    while(hex in list):
        r = randint(0,255)
        b = randint(0,255)
        g = randint(0,255)
        hex = '#%02x%02x%02x' % (r,g,b)
  
    return hex

"""
A função train recebe os parâmetros:

G: o grafo da rede KNN.
algorithm: recebe um string contendo qual algoritmo será usado para o agrupamento.
show_image: plota o grafo usando o matplotlib.

"""
def train(G, dict_classes, algorithm, show_image = False):
    dict_predictclasses = {}
    #Verifica qual é o algoritmo de agrupamento a ser usado.
    if isinstance(algorithm, ParametersLabelPropagation):
        glp = list(nx.community.label_propagation_communities(G))
    elif isinstance(algorithm, ParametersAsynchronousLabelPropagation):
        glp = list(nx.community.asyn_lpa_communities(G, algorithm.weight, algorithm.seed))
    elif isinstance(algorithm, ParametersGreedyModularity):   
        glp = nx.community.greedy_modularity_communities(G, algorithm.weight)
    elif isinstance(algorithm, ParametersGirvanNewman):
        glp = nx.community.centrality.girvan_newman(G, algorithm.most_valuable_edge)
        glp = list(sorted(c) for c in next (glp))
    elif isinstance(algorithm, ParametersEdgeBetweennessCentrality):
        ebc_normalized = algorithm.normalized
        ebc_weight = algorithm.weight
        glp = nx.community.centrality.girvan_newman(G, most_central_edge)
        glp = list(sorted(c) for c in next (glp))
        
    #list_glp representa a lista de todos os documentos.
    list_glp = []
    #colors representa a lista de cores que será usada para plotar  o grafo.
    colors = []
    
    for i in list(glp):
        #Gera uma cor em formato hex para a classe.
        color = generate_and_convert_color(colors)
        #Iterador de cada classe na lista de grupos glp.
        for j in i:
            #Cada elemento da classe terá a mesma cor.
            colors.append(color)
            #Adiciona cada documento a uma lista.
            list_glp.append(j)

            #O documento j terá sua classe prevista alocada no dicionário.
            dict_predictclasses[j] = dict_classes[find_cluster_class(i)]
    layout = nx.spring_layout(G, k = 0.35, scale = 100)
    colormap = []
    
    #A imagem apenas será plotada, se tiver a varíavel show_image ativada via parâmetro.
    if show_image:
        
        if isinstance(algorithm, ParametersLabelPropagation) or isinstance(algorithm, ParametersAsynchronousLabelPropagation):
            #Percorre os elementos do grafo.
            for i in G:
                glp = nx.community.label_propagation_communities(G)
                #Para cada sub-grupo.
                for enum, j  in enumerate(glp):
                  #Para cada documento.
                    for t in j:
                      #Se o elemento for o mesmo na posição do grafo, sua cor será atribuida.
                        if t == i:
                            colormap.append(colors[list_glp.index(t)])

        if isinstance(algorithm, ParametersGreedyModularity):
          #Percorre os elementos do grafo.
            for i in G:
              #Para cada sub-grupo.
                for enum, j  in enumerate(glp):
                  #Para cada documento.
                    for t in j:   
                      #Se o elemento for o mesmo na posição do grafo, sua cor será atribuida.     
                        if t == i:
                            colormap.append(colors[list_glp.index(t)])

            if isinstance(algorithm, ParametersGirvanNewman):
              #Percorre os elementos do grafo.
                for i in G:
                  #Para cada sub-grupo.
                    for enum, j  in enumerate(glp[0]):
                      #Para cada documento.
                        for t in j:   
                          #Se o elemento for o mesmo na posição do grafo, sua cor será atribuida.     
                            if t == i:
                                colormap.append(colors[list_glp.index(t)])

        #Parâmetros do Nx.draw()
        size = 200
        font = 8
        width = 8
        height = 6
        
        nx.draw(G, layout, node_color = colormap, with_labels=False, node_size = size, font_size = font)
        plt.draw() 
        figure = plt.gcf()
        figure.set_size_inches(20,11)
        figure.set_size_inches(width, height)
        #Salva a imagem como graph.png.
        plt.savefig('graph.png', dpi = 100)
    
    lst_correct = list(dict_classes.values())
    #Cada classe do documento n será alocado na lista de elementos corretos.
        
    #Cada classe prevista será alocada na lista de elementos previstos.
    lst_predict= list(dict_predictclasses.values())
    
    #Com base na lista de classes corretas e previstas será gerada uma matrix de confusão.
    carray = confusion_matrix(lst_correct, lst_predict)
    
    #Verifica se a rede é desconexa, se for o resultado não será processado
    if len(list(glp)) < len(np.unique(lst_correct)) or  len(list(glp)) > math.sqrt(len(list_glp)):
      return None, None, None, None, None
    
    """
    Será retornado: 
    A lista de grupos detectados na comunidades,
    Lista de todos os documentos,
    A matriz de confusão,
    A lista de classes corretas,
    A lista de classes previstas.
    """
    
    if isinstance(algorithm, ParametersLabelPropagation):
        glp = nx.community.label_propagation_communities(G)
    elif isinstance(algorithm, ParametersAsynchronousLabelPropagation):
        glp = nx.community.asyn_lpa_communities(G, algorithm.weight, algorithm.seed)
        
    return glp, list_glp, carray, lst_correct, lst_predict
