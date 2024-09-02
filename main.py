import csv
from typing import List
from util import normalize_by_feature_scaling
from network import Network
from random import shuffle

def iris_test(hidden_nodes, learning_rate, training_attempts):
    iris_parameters: List[List[float]] = []
    iris_classifications: List[List[float]] = []
    iris_species: List[str] = []
    with open('iris.csv', mode='r') as iris_file:
        irises: List = list(csv.reader(iris_file))
        shuffle(irises) # get our lines of data in random order
        for iris in irises:
            parameters: List[float] = [float(n) for n in iris[0:4]]
            iris_parameters.append(parameters)
            species: str = iris[4]
            if species == "Iris-setosa":
                iris_classifications.append([1.0, 0.0, 0.0])
            elif species == "Iris-versicolor":
                iris_classifications.append([0.0, 1.0, 0.0])
            else:
                iris_classifications.append([0.0, 0.0, 1.0])
            iris_species.append(species)
    normalize_by_feature_scaling(iris_parameters)

    iris_network: Network = Network([4, hidden_nodes, 3], hidden_nodes)

    def iris_interpret_output(output: List[float]) -> str:
        if max(output) == output[0]:
            return "Iris-setosa"
        elif max(output) == output[1]:
            return "Iris-versicolor"
        else:
            return "Iris-virginica"

    # train over the first 140 irises in the data set 50 times
    iris_trainers: List[List[float]] = iris_parameters[0:140]
    iris_trainers_corrects: List[List[float]] = iris_classifications[0:140]
    for _ in range(training_attempts):
        iris_network.train(iris_trainers, iris_trainers_corrects)

    # test over the last 10 of the irises in the data set
    iris_testers: List[List[float]] = iris_parameters[140:150]
    iris_testers_corrects: List[str] = iris_species[140:150]
    iris_results = iris_network.validate(iris_testers, iris_testers_corrects, iris_interpret_output)
    print(f"{iris_results[0]} correct of {iris_results[1]} = {iris_results[2] * 100}%")

def wine_test(hidden_nodes, learning_rate, training_attempts):
    wine_parameters: List[List[float]] = []
    wine_classifications: List[List[float]] = []
    wine_species: List[int] = []
    with open('wine.csv', mode='r') as wine_file:
        wines: List = list(csv.reader(wine_file, quoting=csv.QUOTE_NONNUMERIC))
        shuffle(wines) # get our lines of data in random order
        for wine in wines:
            parameters: List[float] = [float(n) for n in wine[1:14]]
            wine_parameters.append(parameters)
            species: int = int(wine[0])
            if species == 1:
                wine_classifications.append([1.0, 0.0, 0.0])
            elif species == 2:
                wine_classifications.append([0.0, 1.0, 0.0])
            else:
                wine_classifications.append([0.0, 0.0, 1.0])
            wine_species.append(species)
    normalize_by_feature_scaling(wine_parameters)

    wine_network: Network = Network([13, hidden_nodes, 3], learning_rate)

    def wine_interpret_output(output: List[float]) -> int:
        if max(output) == output[0]:
            return 1
        elif max(output) == output[1]:
            return 2
        else:
            return 3

    # train over the first 150 wines 10 times
    wine_trainers: List[List[float]] = wine_parameters[0:150]
    wine_trainers_corrects: List[List[float]] = wine_classifications[0:150]
    for _ in range(training_attempts):
        wine_network.train(wine_trainers, wine_trainers_corrects)

    # test over the last 28 of the wines in the data set
    wine_testers: List[List[float]] = wine_parameters[150:178]
    wine_testers_corrects: List[int] = wine_species[150:178]
    wine_results = wine_network.validate(wine_testers, wine_testers_corrects, wine_interpret_output)
    print(f"{wine_results[0]} correct of {wine_results[1]} = {wine_results[2] * 100}%")

def hidden_nodes():
    print("Press enter for default")
    while True:
        input_value = input("Number of hidden neurons (Warning too large will cause issues): ")
        if input_value == "":
            return None
        try:
            input_number = int(input_value)
            if input_number < 0:
                print("Can't be negative")
            else:
                return input_number
        except:
            print("Invalid input")

def learning_rate():
    print("Press enter for default")
    while True:
        input_value = input("Learning rate (0.0 - 1.0): ")
        if input_value == "":
            return None
        try:
            input_number = float(input_value)
            if input_number < 0 or input_number > 1.0:
                print("Must be between 0.0 - 1.0")
            else:
                return input_number
        except:
            print("Invalid input")


def training_attempts():
    print("Press enter for default Iris (50) Wine (10)")
    while True:
        input_value = input("Training attempts: ")
        if input_value == "":
            return None
        try:
            input_number = int(input_value)
            if input_number < 0:
                print("Can't be negative")
            else:
                return input_number
        except:
            print("Invalid input")
            
if __name__ == "__main__":
    while True:
        print("Simple Neural Network Program")
        option = input("(I)ris test\n(W)ine test\n(Q)uit\n> ").upper()
        if option == "I":
            hidden = hidden_nodes()
            learning = learning_rate()
            training = training_attempts()
            
            # Defaults
            if hidden is None:
                hidden = 6
            if learning is None:
                learning = 0.3
            if training is None:
                training = 50
            
            iris_test(hidden, learning, training)
            
        elif option == "W":
            hidden = hidden_nodes()
            learning = learning_rate()
            training = training_attempts()
            
            # Defaults
            if hidden is None:
                hidden = 7
            if learning is None:
                learning = 0.9
            if training is None:
                training = 10
            
            wine_test(hidden, learning, training)
            
        elif option == "Q":
            print("cya")
            break
        
        else:
            print("Invalid option")
        