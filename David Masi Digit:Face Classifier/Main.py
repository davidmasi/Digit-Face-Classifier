from models.perceptron import PerceptronClass
from models.NeuralNetwork import NeuralNetworkClass

# Generates the statistics for each algorithm and saves the trained models
if __name__ == '__main__':
    
    nn_classifier = NeuralNetworkClass()
    nn_classifier.run()
    nn_classifier.write()
    
    #perceptron_classifier = PerceptronClass()
    #perceptron_classifier.run()
    #perceptron_classifier.write()    