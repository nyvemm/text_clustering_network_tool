import pandas as pd
import numpy as np
import os
import sys

#Essa função recebe um arquivo CSV de processamento e retorna seu pós processamento.
def postprocessing(path):
    if os.path.isfile(path):
        df = pd.read_csv(path, sep='\t')

        list_algorithm = list(df['Algorithm'])
        list_algorithm = np.unique(list_algorithm)

        list_metrics = []

        #String de retorno
        output = ''
        
        columns = []
        #Lista dos algoritmos
        for algorithm in list_algorithm:
            #Substitui o mnemônico do algoritmo
            pro_algorithm = 'Weighted '  + algorithm[:len(algorithm) - 1] if algorithm.endswith('w') else algorithm
            
            output += str(pro_algorithm) + '\n'
            
            current_df = df.loc[df['Algorithm'] == algorithm]
            columns = current_df.columns.tolist()

            for column_range in range(columns.index('Accuracy'), len(columns)):
                metric = np.mean(current_df[columns[column_range]])
                list_metrics.append((algorithm, 'Mean', columns[column_range], metric))
                output += 'Average ' + str(columns[column_range]) + ' : ' + str(metric) + '\n'
                
            output += "\n" 
            for column_range in range(columns.index('Accuracy'), len(columns)):
                metric = np.std(current_df[columns[column_range]])
                output += 'Standard Deviation ' + str(columns[column_range]) + ' : ' + str(metric) + '\n'
            
            output += "\n" 
            for column_range in range(columns.index('Accuracy'), len(columns)):
                metric = max(current_df[columns[column_range]])
                output += 'Maximum ' + str(columns[column_range]) + ' : ' + str(metric) + '\n'
             
            output += "\n" 
            for column_range in range(columns.index('Accuracy'), len(columns)):
                metric = min(current_df[columns[column_range]])
                output += 'Minimum ' + str(columns[column_range]) + ' : ' + str(metric) + '\n'
                
            output += ('-' * 62) + '\n'
            
        list_metrics.sort(key = lambda x :  x[3], reverse = True)
        metrics = columns[columns.index('Accuracy'):]
        string = []

        for i in list_metrics:  
            if i[2] in metrics:
                string.append('Algorithm with most ' +  i[2] + ' : ' + i[0])
                metrics.remove(i[2])
            if len(metrics) == 0:
                break 

        string.sort()
        for i in string:
            output += str(i) + "\n"
        
        list_metrics.sort(key = lambda x :  x[3], reverse = False)
        metrics = columns[columns.index('Accuracy'):]
        string = []
        
        output += "\n"
        for i in list_metrics:  
            if i[2] in metrics:
                string.append('Algorithm with least ' +  i[2] + ' : ' + i[0])
                metrics.remove(i[2])
            if len(metrics) == 0:
                break 

        string.sort()
        for i in string:
            output += str(i) + "\n"
            
        return output    
    else:
        raise Exception('Invalid File')

args = len(sys.argv)
if args == 3:
    inp = sys.argv[1]
    out = sys.argv[2]

    n = os.path.basename(inp).replace('.csv', '.txt')
    print('path =',n)
    with open(os.path.join(out, n), 'w') as file:
        pp = postprocessing(inp)
        print(pp)
        file.write(pp) 
    
