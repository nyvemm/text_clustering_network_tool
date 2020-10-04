import os
import sys
import pandas as pd

args = len(sys.argv)

#Apenas o caminho e a saida.
if args == 3:
    temp_path = sys.argv[1]
    output = sys.argv[2]

    #Cria um Dataframe vazio.
    big_df = pd.DataFrame()
    #Percorre todos os arquivos do caminho.
    temp_files = os.listdir(temp_path)

    for files in temp_files:
        #Verifica se é um arquivo .csv
        if not files.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(temp_path,files), sep='\t')
        #Apenas concatena as linhas que já não existem
        big_df = pd.concat([big_df, df],ignore_index=True).drop_duplicates().reset_index(drop=True)

    #Concatena o último dataframe com os dataframes temporários
    print(big_df.head)

    #Salva o arquivo em CSV.
    big_df.to_csv(os.path.join(output, 'Merge.csv'),  sep='\t', encoding='utf-8', index = False)
else:
    print('Argumentos inválidos')
        
        