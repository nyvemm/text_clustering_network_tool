import time
import os

from parameters import *
from json_reader import create_json_config

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

def json_batch_config(temp_path, file_path, pr_file_name = None):

    #Sequência de execução padrão dos algoritmos
    alg_seq = ['labelpropagation',
                'asynchronouslabelpropagation',
                'asynchronouslabelpropagationw',
                'greedymodularity',
                'greedymodularityw',
                'girvannewman',
                'edgebetweennesscentrality',
                'edgebetweennesscentralityw']

    temp_files = os.listdir(temp_path)
    empty_temp_files = True

    if(not pr_file_name == None):
        #Cria uma lista para guardar o nome do arquivo e a data de criação
        file_name_ctime = []

        for files in temp_files:
            if files.endswith('.csv'):
                current_time = os.path.getmtime(os.path.join(temp_path, files))
                file_name_ctime.append((files, current_time))
                empty_temp_files = False

    if not empty_temp_files:
        #Ordena a lista pela data de criação do mais antigo para o mais crescente
        file_name_ctime.sort(key = lambda x : x[1])
        
        last_cr_file = file_name_ctime[len(file_name_ctime) - 1][0]

        last_cr_file = last_cr_file.split('_')

        l_fname = last_cr_file[1] + '.csv'
        l_algorithm = alg_seq.index(last_cr_file[2])
        l_k = last_cr_file[3]
    else:
        l_fname = pr_file_name   
        l_algorithm = 0
        l_k = 1
    
    network_parameters = ParametersKNNNetwork('Cosine', [3,5,7,9,11,13,15,17,19,21,23,25])
    #Cria um parâmetro de rede para a rede atual
    range_k = list(range(int(l_k) + 2, 27, 2))

    #Todos os k do algoritmo foram processados
    if len(range_k) == 0:
        l_algorithm += 1
        #É definido os parâmetros padrões
        current_network_parameters = network_parameters
    else:
        current_network_parameters = ParametersKNNNetwork('Cosine', range_k)

    if l_fname == None:
        l_fname = pr_file_name

    f_name = os.path.join(file_path, l_fname)
    if l_algorithm == 0:
        parameters = ParametersLabelPropagation(10)
        create_json_config(f_name, 'Label Propagation', parameters, 'KNN', current_network_parameters, clear = True)
        parameters = ParametersAsynchronousLabelPropagation(False, 10, 10)
        create_json_config(f_name, 'Asynchronous Label Propagation', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersAsynchronousLabelPropagation(True, 10, 10)
        create_json_config(f_name, 'Asynchronous Label Propagation', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersGreedyModularity(False, 10)
        create_json_config(f_name, 'Greedy Modularity', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersGreedyModularity(True, 10)
        create_json_config(f_name, 'Greedy Modularity', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersGirvanNewman(None, 10)
        create_json_config(f_name, 'Girvan Newman', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
    elif l_algorithm == 1:
        parameters = ParametersAsynchronousLabelPropagation(False, 10, 10)
        create_json_config(f_name, 'Asynchronous Label Propagation', parameters, 'KNN', current_network_parameters, clear = True)
        parameters = ParametersAsynchronousLabelPropagation(True, 10, 10)
        create_json_config(f_name, 'Asynchronous Label Propagation', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersGreedyModularity(False, 10)
        create_json_config(f_name, 'Greedy Modularity', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersGreedyModularity(True, 10)
        create_json_config(f_name, 'Greedy Modularity', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersGirvanNewman(None, 10)
        create_json_config(f_name, 'Girvan Newman', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
    elif l_algorithm == 2:
        parameters = ParametersAsynchronousLabelPropagation(True, 10, 10)
        create_json_config(f_name, 'Asynchronous Label Propagation', parameters, 'KNN', current_network_parameters, clear = True)
        parameters = ParametersGreedyModularity(False, 10)
        create_json_config(f_name, 'Greedy Modularity', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersGreedyModularity(True, 10)
        create_json_config(f_name, 'Greedy Modularity', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersGirvanNewman(None, 10)
        create_json_config(f_name, 'Girvan Newman', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
    elif l_algorithm == 3:
        parameters = ParametersGreedyModularity(False, 10)
        create_json_config(f_name, 'Greedy Modularity', parameters, 'KNN', current_network_parameters, clear = True)
        parameters = ParametersGreedyModularity(True, 10)
        create_json_config(f_name, 'Greedy Modularity', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersGirvanNewman(None, 10)
        create_json_config(f_name, 'Girvan Newman', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
    elif l_algorithm == 4:  
        parameters = ParametersGreedyModularity(True, 10)
        create_json_config(f_name, 'Greedy Modularity', parameters, 'KNN', current_network_parameters, clear = True)
        parameters = ParametersGirvanNewman(None, 10)
        create_json_config(f_name, 'Girvan Newman', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
    elif l_algorithm == 5:      
        parameters = ParametersGirvanNewman(None, 10)
        create_json_config(f_name, 'Girvan Newman', parameters, 'KNN', current_network_parameters, clear = True)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
    elif l_algorithm == 6:  
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', current_network_parameters, clear = True)
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', network_parameters, clear = False)
    elif l_algorithm == 7:  
        parameters = ParametersEdgeBetweennessCentrality(True, False, 10)
        create_json_config(f_name, 'Edge Betweenness Centrality', parameters, 'KNN', current_network_parameters, clear = True)



