from helpers import getData

from models.perceptron import PerceptronClass
from models.NaiveBayes import NaiveBayesClass

#Generate Statistics for each classifier.
if __name__ == '__main__':
    
    classifierOne = PerceptronClass()
    classifierOne.run()
    classifierOne.write()

    classifierTwo = NaiveBayesClass()
    classifierTwo.run()
    classifierTwo.write()