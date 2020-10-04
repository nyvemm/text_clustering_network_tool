#Classe de Parâmetros do Algoritmo Label Propagation.
class ParametersLabelPropagation:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations

#Classe de Parâmetros do Algoritmo Asynchronous Label Propagation.
class ParametersAsynchronousLabelPropagation:
    def __init__(self, weight, seed, max_iterations):
        self.weight = weight
        self.seed = seed
        self.max_iterations = max_iterations
    
#Classe de Parâmetros do Algoritmo Greedy Modularity.
class ParametersGreedyModularity:
    def __init__(self, weight, max_iterations):
        self.weight = weight
        self.max_iterations = max_iterations

#Classe de Parâmetros do Algoritmo Girvan Newman.
class ParametersGirvanNewman:
    def __init__(self, most_valuable_edge, max_iterations):
        self.most_valuable_edge = most_valuable_edge
        self.max_iterations = max_iterations

class ParametersEdgeBetweennessCentrality:
    def __init__(self, normalized, weight, max_iterations):
        self.normalized = normalized
        self.weight = weight
        self.max_iterations = max_iterations
        
#Classe de Parâmetros do Algoritmo gerador da rede KNN.
class ParametersKNNNetwork:
    def __init__(self, proximity_measure, number_of_neighbours):
        self.proximity_measure = proximity_measure
        self.number_of_neighbours = number_of_neighbours