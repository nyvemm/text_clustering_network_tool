import json
import pandas as pd
import os.path
import os
import time
import shutil

from experiment_email import sendEmail
from pandas.io.json import json_normalize
from external_evaluation_metrics import *
from cosine import *
from knn_network import *
from parameters import *

#A função create_json_config é utilizada para criar um arquivo .json
def create_json_config(algorithm, parameters, network_type, network_parameters, data, clear = True):
    if clear == True or (not data):
        data = {}
        data['config'] = []
        #Verifica se o algoritmo é o Label Propagation.
        if(isinstance(parameters, ParametersLabelPropagation)):
          #Verifica se a rede a ser gerada é KNN.
            if(isinstance(network_parameters, ParametersKNNNetwork)):
                data['config'].append({
                'algorithm': algorithm,
                'max_iterations': parameters.max_iterations,
                'network_type': network_type,
                'proximity_measure': network_parameters.proximity_measure,
                'number_of_neighbours': network_parameters.number_of_neighbours})
        #Verifica se o algoritmo é o Greedy Modularity.        
        elif(isinstance(parameters, ParametersGreedyModularity)):
          #Verifica se a rede a ser gerada é KNN.
            if(isinstance(network_parameters, ParametersKNNNetwork)):
                data['config'].append({
                'algorithm': algorithm,
                'max_iterations': parameters.max_iterations,
                'network_type': network_type,
                'proximity_measure': network_parameters.proximity_measure,
                'number_of_neighbours': network_parameters.number_of_neighbours})
        elif(isinstance(parameters, ParametersAsynchronousLabelPropagation)):
          #Verifica se a rede a ser gerada é KNN.
            if(isinstance(network_parameters, ParametersKNNNetwork)):
                data['config'].append({
                'algorithm': algorithm,
                'weight': parameters.weight,
                'seed' : parameters.seed, 
                'max_iterations': parameters.max_iterations,
                'network_type': network_type,
                'proximity_measure': network_parameters.proximity_measure,
                'number_of_neighbours': network_parameters.number_of_neighbours})
        elif(isinstance(parameters, ParametersGirvanNewman)):
          #Verifica se a rede a ser gerada é KNN.
            if(isinstance(network_parameters, ParametersKNNNetwork)):
                data['config'].append({
                'algorithm': algorithm,
                'most_valuable_edge': parameters.most_valuable_edge,
                'max_iterations': parameters.max_iterations,
                'network_type': network_type,
                'proximity_measure': network_parameters.proximity_measure,
                'number_of_neighbours': network_parameters.number_of_neighbours})
        elif(isinstance(parameters, ParametersEdgeBetweennessCentrality)):
          #Verifica se a rede a ser gerada é KNN.
            if(isinstance(network_parameters, ParametersKNNNetwork)):
                data['config'].append({
                'algorithm': algorithm,
                'normalized': parameters.normalized,
                'weight': parameters.weight,
                'max_iterations': parameters.max_iterations,
                'network_type': network_type,
                'proximity_measure': network_parameters.proximity_measure,
                'number_of_neighbours': network_parameters.number_of_neighbours})
    else:
        #Verifica se o algoritmo é o Label Propagation.
        if(isinstance(parameters, ParametersLabelPropagation)):
          #Verifica se a rede a ser gerada é KNN.
            if(isinstance(network_parameters, ParametersKNNNetwork)):
                data['config'].append({
                'algorithm': algorithm,
                'max_iterations': parameters.max_iterations,
                'network_type': network_type,
                'proximity_measure': network_parameters.proximity_measure,
                'number_of_neighbours': network_parameters.number_of_neighbours})
        #Verifica se o algoritmo é o Greedy Modularity.        
        elif(isinstance(parameters, ParametersGreedyModularity)):
          #Verifica se a rede a ser gerada é KNN.
            if(isinstance(network_parameters, ParametersKNNNetwork)):
                data['config'].append({
                'algorithm': algorithm,
                'weight': parameters.weight,
                'max_iterations': parameters.max_iterations,
                'network_type': network_type,
                'proximity_measure': network_parameters.proximity_measure,
                'number_of_neighbours': network_parameters.number_of_neighbours})
                #Verifica se o algoritmo é o Greedy Modularity.        
        elif(isinstance(parameters, ParametersAsynchronousLabelPropagation)):
          #Verifica se a rede a ser gerada é KNN.
            if(isinstance(network_parameters, ParametersKNNNetwork)):
                data['config'].append({
                'algorithm': algorithm,
                'weight': parameters.weight,
                'seed' : parameters.seed, 
                'max_iterations': parameters.max_iterations,
                'network_type': network_type,
                'proximity_measure': network_parameters.proximity_measure,
                'number_of_neighbours': network_parameters.number_of_neighbours})
        elif(isinstance(parameters, ParametersGirvanNewman)):
          #Verifica se a rede a ser gerada é KNN.
            if(isinstance(network_parameters, ParametersKNNNetwork)):
                data['config'].append({
                'algorithm': algorithm,
                'most_valuable_edge': parameters.most_valuable_edge,
                'max_iterations': parameters.max_iterations,
                'network_type': network_type,
                'proximity_measure': network_parameters.proximity_measure,
                'number_of_neighbours': network_parameters.number_of_neighbours})
        elif(isinstance(parameters, ParametersEdgeBetweennessCentrality)):
          #Verifica se a rede a ser gerada é KNN.
            if(isinstance(network_parameters, ParametersKNNNetwork)):
                data['config'].append({
                'algorithm': algorithm,
                'normalized': parameters.normalized,
                'weight': parameters.weight,
                'max_iterations': parameters.max_iterations,
                'network_type': network_type,
                'proximity_measure': network_parameters.proximity_measure,
                'number_of_neighbours': network_parameters.number_of_neighbours})
    return data

"""
A função read_and_execute_json_config recebe o caminho de um arquivo de
configuração .json e cria um Dataframe contendo as medidas de avaliação
externa.

stdout: indica se a média das medidas de avaliação externa serão calculadas
e exibidas na tela.
"""
def read_and_execute_json_config(file_path, data = {}, temp = '', stdout = False, printable = False, delete_temp = False):
    if temp and not os.path.isdir(temp):
        os.makedirs(temp)

    #Result Dictionary
    rd = {'Base Name':[], 
            #Algorithm Parameters
            'Algorithm':[],
            'Weight':[],
            'Seed':[],
            'Most Valuable Edge':[],
            'Normalized':[],
            #Network Parameters
            'Network Type':[],
            'K':[],
            'Proximity Measure':[],
            #Metrics
            'Repetition':[],
            'Accuracy':[],
            'Purity':[],
            'Micro-Precision':[],
            'Macro-Precision':[],
            'Micro-Recall':[],  
            'Macro-Recall':[],
            'Micro-F1-Score':[],
            'Macro-F1-Score':[],
            'Entropy':[],
            #Sklearn.metrics
            'Adjusted Rand':[],
            'Completeness':[],
            'Fowlkes Mallows':[],
            'Homogeneity':[],
            'Mutual Info':[],
            'V Measure':[],
            'Model Building Time':[] 
            }
    
    #Verifica se há mais de um arquivo de configuração.
    for i in data['config']:
        #Lê o documento.
        df = pd.read_csv(file_path, sep=',')
        #Pega apenas a base do diretório absoluto.
        base_name = os.path.basename(file_path)
        
        #Parâmetros dos Algoritmos
        k = i.get('k', None)
        max_iterations = i.get('max_iterations', 10)
        most_valuable_edge = i.get('most_valuable_edge', None)
        normalized = i.get('normalized', True)
        seed = i.get('seed', max_iterations)
        weight = i.get('weight', None)
        weight = 'weight' if weight == True else weight
        
        #Parâmetros das Redes
        proximity_measure = i.get('proximity_measure', 'Cossine')
        number_of_neighbours = i.get('number_of_neighbours', list(range(3,26,2)))

        #Verifica se tem mais de um k, a ser gerado na rede KNN.
        if isinstance(number_of_neighbours, int):
            number_of_neighbours = [i['number_of_neighbours']]
            
        #Verifica qual é o algoritmo de agrupamento e cria uma instância de classe.
        if i['algorithm'] == 'Label Propagation':
            algorithm = ParametersLabelPropagation(max_iterations)
        if i['algorithm'] == 'Asynchronous Label Propagation':
            algorithm = ParametersAsynchronousLabelPropagation(weight, seed, max_iterations)
        if i['algorithm'] == 'Greedy Modularity':
            algorithm = ParametersGreedyModularity(weight, max_iterations)
        if i['algorithm'] == 'Girvan Newman':
            algorithm = ParametersGirvanNewman(most_valuable_edge, max_iterations)
        if i['algorithm'] == 'Edge Betweenness Centrality':
            algorithm = ParametersEdgeBetweennessCentrality(normalized, weight, max_iterations)
        
        #Verifica qual é o algoritmo de criação de rede e cria uma instância de classe
        if i['network_type'] == 'KNN':
            network_type = ParametersKNNNetwork(proximity_measure, number_of_neighbours) 
        
        #Loop para cada k da rede KNN.
        if printable:
            print('\nAlgorithm =', i['algorithm'])
        for ki in network_type.number_of_neighbours:
            if printable:
                print('\nK = ', ki)
            if(stdout):    
                avg_accuracy = 0
                avg_purity = 0
                avg_micro_precision = 0
                avg_macro_precision = 0
                avg_micro_recall = 0
                avg_macro_recall = 0
                avg_micro_f1_score = 0
                avg_macro_f1_score = 0
                avg_entropy = 0
                #Sklearn.metrics
                avg_adjusted_rand = 0
                avg_completeness = 0
                avg_fowlkes_mallows = 0
                avg_homogeneity = 0
                avg_mutual_info = 0
                avg_v_measure = 0
                avg_model_building_time = 0

            #Calcula a similharidade pelo cosseno
            list_cosseno, dict_classes = similarity_values(df, ki)
            #Calcula a rede KNN
            if(weight == 'weight'):
                G = knn_network(list_cosseno, dict_classes, k = ki, weighted = True, limit = -1)
            else:
                G = knn_network(list_cosseno, dict_classes, k = ki, limit = -1)
            #Percorre um número de iterações pré-definido no arquivo de configuração.
            for j in range(algorithm.max_iterations):  
                if printable:
                    print('Repetition = ', j)
                starttime = time.time()
                
                #Treina o algoritmo
                glp, list_glp, carray, lst_correct, lst_predict = train(G, dict_classes, algorithm, show_image = False)
                
                #Calcula o resultado apenas se números de classes <= número de grupos <= raiz do número de documentos
                disjointed = lst_correct == None or lst_predict == None
                
                #Se a rede for desconexa então o processamento será itnterrompido
                if(disjointed):
                    break
                #As medidas de validação externa são calculadas.
                m_accuracy, m_purity, m_micro_precision, m_macro_precision, m_micro_recall, m_macro_recall, m_micro_f1_score, m_macro_f1_score, m_entropy, m_adjusted_rand, m_completeness, m_fowlkes_mallows, m_homogeneity, m_mutual_info, m_v_measure = external_evaluate_metrics(carray, glp, list_glp, lst_correct, lst_predict, dict_classes)
                    
                m_model_building_time = time.time() - starttime

                #Calcula a média das medidas de avaliação externa
                if(stdout):    
                    avg_accuracy += m_accuracy
                    avg_purity += m_purity
                    avg_micro_precision += m_micro_precision
                    avg_macro_precision += m_macro_precision
                    avg_micro_recall += m_micro_recall
                    avg_macro_recall += m_macro_recall
                    avg_micro_f1_score += m_micro_f1_score
                    avg_macro_f1_score += m_macro_f1_score
                    avg_entropy += m_entropy
                    #Sklearn.metrics
                    avg_adjusted_rand += m_adjusted_rand
                    avg_completeness += m_completeness
                    avg_fowlkes_mallows += m_fowlkes_mallows
                    avg_homogeneity += m_homogeneity
                    avg_mutual_info += m_mutual_info
                    avg_v_measure += m_v_measure
                    avg_model_building_time += m_model_building_time

                rd['Base Name'].append(base_name)
                rd['Algorithm'].append(i['algorithm'])


                if isinstance(algorithm, ParametersLabelPropagation):
                    rd['Weight'].append('-')
                    rd['Seed'].append('-')
                    rd['Most Valuable Edge'].append('-')
                    rd['Normalized'].append('-')
                if isinstance(algorithm, ParametersAsynchronousLabelPropagation):
                    rd['Weight'].append(str(algorithm.weight))
                    rd['Seed'].append(str(algorithm.seed))
                    rd['Most Valuable Edge'].append('-')
                    rd['Normalized'].append('-')
                elif isinstance(algorithm, ParametersGreedyModularity):
                    rd['Weight'].append(str(algorithm.weight))
                    rd['Seed'].append('-')
                    rd['Most Valuable Edge'].append('-')
                    rd['Normalized'].append('-')
                elif isinstance(algorithm, ParametersGirvanNewman):
                    rd['Weight'].append('-')
                    rd['Seed'].append('-')
                    rd['Most Valuable Edge'].append(str(algorithm.most_valuable_edge))
                    rd['Normalized'].append('-')
                elif isinstance(algorithm, ParametersEdgeBetweennessCentrality):
                    rd['Normalized'].append(str(algorithm.normalized))
                    rd['Weight'].append(str(algorithm.weight))
                    rd['Seed'].append('-')
                    rd['Most Valuable Edge'].append('-')

                rd['Network Type'].append( i['network_type'])
                if isinstance(network_type, ParametersKNNNetwork):
                    rd['K'].append(ki)
                    rd['Proximity Measure'].append(network_type.proximity_measure)

                rd['Repetition'].append(j)
                rd['Accuracy'].append(m_accuracy)
                rd['Purity'].append(m_purity)
                rd['Micro-Precision'].append(m_micro_precision)
                rd['Macro-Precision'].append(m_macro_precision)
                rd['Micro-Recall'].append(m_micro_recall)
                rd['Macro-Recall'].append(m_macro_recall)
                rd['Micro-F1-Score'].append(m_micro_f1_score)
                rd['Macro-F1-Score'].append(m_macro_f1_score)
                rd['Entropy'].append(m_entropy)
                #Sklearn.metrics
                rd['Adjusted Rand'].append(m_adjusted_rand)
                rd['Completeness'].append(m_completeness)
                rd['Fowlkes Mallows'].append(m_fowlkes_mallows)
                rd['Homogeneity'].append(m_homogeneity)
                rd['Mutual Info'].append(m_mutual_info)
                rd['V Measure'].append(m_v_measure)
                rd['Model Building Time'].append(m_model_building_time)

            #Se a média for calculada, ela será exibida na tela.
            if(stdout and not disjointed):     
                print('\n------------------------------------------')
                print('Average Accuracy = ' , avg_accuracy / algorithm.max_iterations)
                print('Average Purity = ' , avg_purity / algorithm.max_iterations)
                print('Average Micro-Precision = ' , avg_micro_precision / algorithm.max_iterations)
                print('Average Macro-Precision = ' , avg_macro_precision / algorithm.max_iterations)
                print('Average Micro-Recall = ' , avg_micro_recall / algorithm.max_iterations)
                print('Average Macro-Recall = ' , avg_macro_recall / algorithm.max_iterations)
                print('Average Micro-F1-Score = ' , avg_micro_f1_score / algorithm.max_iterations)
                print('Average Macro-F1-Score = ' , avg_macro_f1_score / algorithm.max_iterations)
                print('Average Entropy = ' , avg_entropy / algorithm.max_iterations)
                #Sklearn.metrics
                print('Average Adjusted Rand = ' , avg_adjusted_rand / algorithm.max_iterations)
                print('Average Completeness = ' , avg_macro_f1_score / algorithm.max_iterations)
                print('Average Fowlkes Mallows = ' , avg_macro_f1_score / algorithm.max_iterations)
                print('Average Homogeneity = ' , avg_macro_f1_score / algorithm.max_iterations)
                print('Average Mutual Info = ' , avg_macro_f1_score / algorithm.max_iterations)
                print('Average V Measure = ' , avg_macro_f1_score / algorithm.max_iterations)
                print('Average Model Building Time = ' , avg_model_building_time / algorithm.max_iterations, '\n')

            #Salva os arquivos temporarios
            rd['Weight'] = [True if v is 'weight' else v for v in rd['Weight']]
            rd['Weight'] = ['-' if v is 'None' else v for v in rd['Weight']]
            #Retira os None dos dicionário e os substitui por '-'
            rd['Weight'] = ['-' if v is None else v for v in rd['Weight']]
            rd['Seed'] = ['-' if v is None else v for v in rd['Seed']]
            rd['Seed'] = ['-' if v is 'None' else v for v in rd['Seed']]
            rd['Most Valuable Edge'] = ['-' if v is None else v for v in rd['Most Valuable Edge']]
            rd['Most Valuable Edge'] = ['-' if v is 'None' else v for v in rd['Most Valuable Edge']]
            rd['Normalized'] = ['-' if v is None else v for v in rd['Normalized']]
            rd['Normalized'] = ['-' if v is 'None' else v for v in rd['Normalized']]

            rd['K'] = ['-' if v is None else v for v in rd['K']]
            rd['Proximity Measure'] = ['-' if v is None else v for v in rd['Proximity Measure']]

            newDF = pd.DataFrame(rd)
    
            base_name_weight = ''
            if hasattr(algorithm, 'weight'):
                if algorithm.weight == 'weight' or algorithm.weight == True:
                    base_name_weight = 'w'

            if temp:
                name = 'temp_' + base_name[:base_name.rindex('.')] + '_' + i['algorithm'].lower().replace(" ", "") + base_name_weight + '_' + str(ki) + '.csv'
                file_name = os.path.join(temp, name)
                newDF.to_csv(file_name, sep='\t', encoding='utf-8', header = True, index = False)
    
    #Transforma o dicionário em um DataFrame.
    newDF = pd.DataFrame(rd)
    if temp:
        newDF = merge_dataframes(temp, newDF)
        
        if delete_temp:
            shutil.rmtree(temp)
    
    return newDF

def merge_dataframes(temp_path, dataframe):
    big_df = pd.DataFrame()
    temp_files = os.listdir(temp_path)
    for files in temp_files:
        #Verifica se é um arquivo .csv
        if not files.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(temp_path,files), sep='\t')
        #Apenas concatena as linhas que já não existem
        big_df = pd.concat([big_df, df],ignore_index=True).drop_duplicates().reset_index(drop=True)

    #Concatena o último dataframe com os dataframes temporários
    big_df = pd.concat([big_df, dataframe],ignore_index=True).drop_duplicates().reset_index(drop=True)
    return big_df

def json_batch_config(temp_path, pr_file_name = None, json_path='config.json'):

    #Sequência de execução padrão dos algoritmos
    alg_seq = ['labelpropagation',
                'asynchronouslabelpropagation',
                'asynchronouslabelpropagationw',
                'greedymodularity',
                'greedymodularityw',
                'girvannewman',
                'edgebetweennesscentrality',
                'edgebetweennesscentralityw']

    json_file = open(json_path,'r')
    json_data = json_file.read()
    json_data = json.loads(json_data)
    json_file.close()
 
    #Lista os arquivos do diretório temporário
    if os.path.isdir(temp_path):
        temp_files = os.listdir(temp_path)
    else:
        temp_files = []
        empty_temp_files = True

    #Conta os k imediatos
    algorithm_count_k = {}

    for algorithm in json_data['config']:
        name = algorithm.get('algorithm')
        weight = algorithm.get('weight', None)

        if name == 'Label Propagation':
            name = 'labelpropagation'
        elif name == 'Asynchronous Label Propagation':
            if not weight:
                name = 'asynchronouslabelpropagation'
            else:
                name = 'asynchronouslabelpropagationw'
        elif name =='Greedy Modularity':
            if not weight:
                name = 'greedymodularity'
            else:
                name = 'greedymodularityw'
        elif name == 'Girvan Newman':
            name = 'girvannewman'
        elif name == 'Edge Betweenness Centrality':
            if not weight:
                name = 'edgebetweennesscentrality'
            else:
                name = 'edgebetweennesscentralityw'

        algorithm_count_k[name] = algorithm['number_of_neighbours']

    if(not pr_file_name == None):
        #Cria uma lista para guardar o nome do arquivo e a data de criação

        #Percorre a lista dos arquivos temporários
        for files in temp_files:
            if files.endswith('.csv') and files.startswith('temp') and pr_file_name.lower() in files.lower():
                split_files = files.split('_')
                try:
                    algorithm_count_k[split_files[2]].remove(int(split_files[3][:-4]))
                except:
                    pass
    
    for i in json_data['config']:
        algorithm['number_of_neighbours'] = algorithm_count_k.pop(list(algorithm_count_k.keys())[0])
    #print(json.dumps(json_data, indent=4, sort_keys=True))

    return json_data
