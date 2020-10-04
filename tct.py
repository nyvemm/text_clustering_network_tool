import sys
import os
import json

from json_reader import *
import experiment_email
import socket

def compute(path, max_size = -1, output = 'Output/', temp=False, temp_dir='', mail=False , del_temp = False):
    if not os.path.isdir(output):
        os.mkdir(output)
    
    #Verifica se os arquivos temporários vão ser utilizados
    if temp:
        #Verifica o diretório informado já foi criado
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)          
    else:
        temp_dir = ''
    
    output_f = list(os.listdir(output))
    
    if temp:
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)

    for f_name in os.listdir(path):
        #Verifica se o tamanho do arquivo é menor que o tamanho especificado
        size = os.path.getsize(os.path.join(path,f_name)) / (10 ** 6)
        if size <= max_size or max_size == -1:
            file = os.path.basename(f_name)
            #Verifica se já o arquivo já foi processado
            if file not in output_f:
                print('\n', file, sep ='')
                current_temp = os.path.join(temp_dir, file[:-4])
                #Se o novo diretório temporário não existir, ele será criado
                if not os.path.isdir(current_temp):
                    os.mkdir(current_temp)
                data = json_batch_config(current_temp, pr_file_name = file[:-4], json_path = settings['config'])
                Dff = read_and_execute_json_config(os.path.join(path,f_name), data = data, temp = current_temp, stdout = False, printable = True, delete_temp = del_temp)
                file_name = os.path.join(output, file)
                Dff.to_csv(file_name, sep='\t', encoding='utf-8', header = True, index = False)
                if mail:
                    experiment_email.sendEmail(socket.gethostname().title(), file, file_name)

#Carrega o arquivo de configuração
with open('settings.json') as file:
    settings = json.loads(file.read())

args = len(sys.argv)
#Verifica se tem um argumento
if args > 1:
    #Verifica se tem apenas um argumento
    if args != 2:
        print('Invalid arguments')
    else:
        #É um diretório
        if os.path.isdir(sys.argv[1]):
            #Verifica se vai ou não usar o caminho padrão
            if not settings['default_output_path']:
                settings['default_output_path'] = 'Output/'
            if not settings['temp_files']:
                settings['default_temp_path'] = 'Output/temp/'
            if not settings['config']:
                settings['config'] = 'config.json'
            if not settings['delete_temp_folder']:
                settings['delete_temp_folder'] = False

            compute(sys.argv[1], max_size = settings['batch_files_max_size'], output = settings['default_output_path'], temp = settings['temp_files'], temp_dir=settings['default_temp_path'], mail=settings['send_mail'], del_temp=settings['delete_temp_folder'])
        #É um arquivo
        elif os.path.isfile(sys.argv[1]):
            if str(sys.argv[1]).endswith('.csv'):
                file = os.path.basename(sys.argv[1][:-4])
                #Cria uma pasta para o nome do arquivo temporário.
                current_temp = os.path.join(settings['default_temp_path'], file)
                if not os.path.isdir(current_temp):
                    os.mkdir(current_temp)

                output_path = os.path.join(settings['default_output_path'], os.path.basename(sys.argv[1]))
                data = json_batch_config(current_temp, pr_file_name = file, json_path = settings['config'])
                Dff = read_and_execute_json_config(sys.argv[1], data = data, stdout = False, printable = True, temp=current_temp, delete_temp = settings['delete_temp_folder'])
                Dff.to_csv(output_path, sep='\t', encoding='utf-8', index = False)
                if settings['send_mail']: 
                    experiment_email.sendEmail(socket.gethostname().title(), file, output_path)
            else:
                print('The file is not a JSON file')
        else:
            print('Invalid arguments')
else:
    print('No arguments')
