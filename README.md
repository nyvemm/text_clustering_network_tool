This is a framework of algorithms for detecting communities in text. It was implemented in Python using the [NetworkX](https://networkx.github.io/) and [scikit-learn](https://scikit-learn.org/) libraries.

# Overview
-----------
The major features of this *framework* include the:

* **Implementation of community detection algorithms**
  * Label Propagation Algorithm
  * Greedy Modularity Algorithm
  * Girvan Newman Algorithm
  * Edge Betweeness Algorithm (ULRIK, 2008)
* **Results processed separately and then merged**
* **Sending e-mails to indicate the end of processing**

# Requeriments
-----------

The codebase is implemented in Python 3.7.9 (64-bit). Package versions used for development are just below.
```
networkx          2.5
numpy             1.19.0
pandas            1.1.2
scikit-learn      0.23.2
```

# Datasets
---------
For the experimental evaluations, we used 21 text collections from different domains, these datasets are available at: [Sequence of words](https://github.com/ragero/text-collections/tree/master/Sequence_of_words_CSV).

# Options
---------
The cluster model is handled by the **tct.py** script which provides the following command line arguments.

```
  --filename        STR        directory of the CSV file to be processed       
```

## Settings file

The setting file is defined by settings.json and provides the following adjustment options.

```
temp_files                  BOOL        defines if the temporary files will be generated to resume execution afterwards
default_output_path         STR         defines the directory of the output files
default_temp_path           STR         defines the directory of the temp files
batch_files_max_size        INT         defines a maximum file size to be processed in a folder
send_mail                   BOOL        defines if the e-mail will be sent after the execution of an experiment
screen_results              BOOL        defines whether the results will be displayed in the output console
config                      STR         defines the default configuration file that defines the algorithms and their parameters to be executed
delete_temp_folder          BOOL        defines if the temporary files folder will be deleted after the experiments are finished
     
```

Each configuration file receives a JSON file containing the algorithms and parameters to be executed and provides the following adjustment options.

```
network_type               STR        defines the type of networks to be generated
proximity_measure          STR        defines the measurement of distances used in the kNN network
number_of_neighbours       LIST       defines a list of k values to be executed on the k-NN network
algorithm                  STR        defines the name of the algorithm to be executed
weight                     BOOL        defines if the algorithm used will include the weights of each network relation
max_iterations             INT        defines the maximum number of iterations of each algorithm to be executed
```
For more specific algorithm parameters, the standard parameters of the [NetworkX](https://networkx.github.io/).

# Results
-------

The results can be found at [Clustering Algorihms](https://nyvemm.github.io/results_clustering_algorithm_network/csv/table.html). For more information, how the methodology used to obtain these results is found in the article.

> Label Propagation in Networks for Text Clustering. Sawada and Rossi, 2018 [[Paper]](https://nyvemm.github.io/results_clustering_algorithm_network).

# Examples

-----

Here is an example of using the framework to process a text collection:

```sh
$ python tct.py CSTR.csv
```

To execute the files in batch, you can pass the directory of the files to be processed:

```sh
$ python tct.py term-frequency
```
