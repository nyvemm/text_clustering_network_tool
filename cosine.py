import pandas as pd
import math

#A função cosine recebe dois dicionários e retorna a similiaridade do coseno deles.
def cosine(dict1, dict2):
  numerador = 0
  norm1 = 0
  norm2 = 0
  for key1 in dict1.keys():
    if key1 in dict2: 
      numerador += dict1[key1] * dict2[key1]
    norm1 += dict1[key1]**2

  for key2 in dict2.keys():
    norm2 += dict2[key2]**2

  cosine = numerador / (math.sqrt(norm1) * math.sqrt(norm2))
  return cosine

#Cria-se um dicionario de classes, alocando cada indíce de um documento a uma classe.
def similarity_values(df, k):
    dict_classes = {}
    contador = 0
    """ 
    O dataframe do arquivo CSV está no seguinte formato:
    TEXT  CLASS
     """
    #Um iterador percorre as tuplas contendo os documentos.
    for row in df.itertuples(index = False):
        #Cada classe é alocada em um dicionário de classes, aonde cada indíce terá sua classe.
        dict_classes[contador] = row[1]
        contador = contador + 1

    """
    Cria-se uma lista de dicionários, para cada documento será criado um dicionário contendo
    a contagem a palavras e a quantidade de ocorrências em um documento, esse dicionári, então
    entrará na lista de dicionários.
    """

    lista_dicts = []
    for i in df['Text'].tolist():
        dados = i.split()
        dict_ex = {}
        for j in dados:
            if j in dict_ex:
                valor = dict_ex[j]
                valor += 1
                dict_ex[j] = valor
            else:
                dict_ex[j] = 1
        lista_dicts.append(dict_ex)

    """
    Após descobrir as palavras e suas ocorrências o valor será salvo em um arquivo .CSV, aonde
    cada arquivo representará um documento, o seu nome será o nome da classe concatenado com 
    sua posição do arquivo original.  
    
    Será calculado a similiaridade do coseno de um documento n para todos os
    outros documentos com exceção dele mesmo, esses valores serão alocados
    em uma lista e depois ordenados pelos valores da similiaridade, gerando
    uma lista com os vizinhos mais similares para os menos vizinhos menos
    similares, após isso essa lista será alocada em outra lista contendo
    as listas para cada documento.
    """

    #Cria uma lista de todos os valores de cosseno
    list_cosseno = []
    for index1, key1 in enumerate(lista_dicts):
        cosseno_atual = []
        for index2, key2 in enumerate(lista_dicts):
            if(index1 != index2):
                if len(cosseno_atual) > k:
                    cosseno_atual.sort(key=lambda tup: tup[1], reverse=True)
                    cosseno_atual.pop(len(cosseno_atual) -1)
                cosseno_atual.append((index2, cosine(key1, key2)))
        #Ordena a lista dos valores da similiaridade do cosseno.
        cosseno_atual.sort(key=lambda tup: tup[1], reverse=True)
        #Adiciona a lista das similharidades de uma documento na lista de cossenos
        list_cosseno.append(cosseno_atual)
    return list_cosseno, dict_classes