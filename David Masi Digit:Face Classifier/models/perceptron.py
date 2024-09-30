import numpy as np
import time
from helpers import featureExtractor, constants, getData, util, statisticsWriter

class PerceptronClass:
    
    def __init__(self):
        self.rawTrainingData = None
        self.trainingLabels = None
        self.rawTestData = None
        self.testLabels = None
        self.legalLabels = None
        self.trainingData = None
        self.testData = None
        self.weights = {}
        self.testIters = 5
        self.statistics = {}
    
    def getFeatures(self, dataType = 'd'):
        if dataType == 'd':
            self.trainingData = list(map(featureExtractor.Digit, self.rawTrainingData))
            self.testData = list(map(featureExtractor.Digit, self.rawTestData))
        elif dataType == 'f':
            self.trainingData = list(map(featureExtractor.Face, self.rawTrainingData))
            self.testData = list(map(featureExtractor.Face, self.rawTestData))
        return True
    
    def classifier(self, data):
        preds = []
        for datum in data:
            cols = util.Counter()
            for label in self.legalLabels:
                cols[label] = self.weights[label] * datum
            preds.append(cols.argMax())
        return preds
    
    def train(self, iterations=3):
        for iteration in range(iterations):
            for index in range(len(self.trainingData)):
                y = self.trainingLabels[index]
                pred = self.classifier([self.trainingData[index]])[0]
                
                if pred != y:
                    self.weights[pred] -= self.trainingData[index]
                    self.weights[y] += self.trainingData[index]
                    
    def test(self):
        self.predictions = self.classifier(self.testData)
        return self.predictions
    
    def run(self, iters = 5, debug=False):
        self.testIters = iters
        if debug:
            TRAINDATA_SIZE = [1.0]
        else:
            TRAINDATA_SIZE = np.arange(0.1, 1.1, 0.1)
        dataTypes = ['d', 'f'] # Digits and faces
        for dataType in dataTypes:
            self.statistics[dataType] = {}
            if dataType == 'f':
                dataSize = constants.FACE_TRAINING_DATA_SIZE
            else:
                dataSize = constants.DIGITS_TRAINING_DATA_SIZE
            for size in TRAINDATA_SIZE:
                acc = []
                avgTime = []
                for index in range(self.testIters):
                    trainingSize = int(size * dataSize)
                    # Fetch data
                    [self.rawTrainingData, 
                    self.trainingLabels,  
                    self.rawTestData, 
                    self.testLabels,
                    self.legalLabels] = getData.fetch(dataType, trainingSize)
                    # Initialize weights counter
                    for label in self.legalLabels:
                        self.weights[label] = util.Counter()
                    # Convert raw data into features
                    self.getFeatures(dataType=dataType)
                    print(f'Perceptron training with {dataType} data and size {trainingSize} [{int(size*100)}%] and iteration {index}...')
                    testStart = time.time()
                    self.train()
                    testTime = time.time() -  testStart
                    avgTime.append(testTime)
                    print(f'Perceptron testing with {dataType} data and size {trainingSize} [{int(size*100)}%] and iteration {index}...')
                    preds = self.test()
                    acc.append([preds[i] == self.testLabels[i] for i in range(len(self.testLabels))].count(True) / len(self.testLabels))
                    print(f'Perceptron prediction accuracy with {dataType} data and size {trainingSize} [{int(size*100)}%]: {acc[index] * 100}% (iteration {index})')
                # Once finished all iterations
                acc = np.array(acc)
                avgTime = np.array(avgTime)
                self.statistics[dataType][int(size*100)] = {}
                self.statistics[dataType][int(size*100)]['mean'] = np.mean(acc)
                self.statistics[dataType][int(size*100)]['std'] = np.std(acc)
                self.statistics[dataType][int(size*100)]['avgTime'] = np.mean(avgTime)
            print()
        return self.statistics
    def write(self):
        statisticsWriter.write(self.statistics, self.testIters)
        
# Testing process
if __name__ == '__main__':
    classifierOne = PerceptronClass()
    classifierOne.run(debug=True)
    classifierOne.write()