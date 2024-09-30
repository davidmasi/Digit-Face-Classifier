import numpy as np
import time
from scipy import optimize as opt
from helpers import featureExtractor, constants, getData, util, statisticsWriter

class NeuralNetworkClass:
    
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
        self.input = None  # without bias node
        self.hidden = None  # without bias node
        self.output = None
        self.dataNum = None
        self.l = None
        "allocate memory for activation matrix of 1s"
        self.inputActivation = np.ones((self.input + 1, self.dataNum)) if self.input and self.dataNum else None
        self.hiddenActivation = np.ones((self.hidden + 1, self.dataNum)) if self.hidden and self.dataNum else None
        self.outputActivation = np.ones((self.output, self.dataNum)) if self.output and self.dataNum else None
        "allocate memory for bias vector"
        self.bias = np.ones((1, self.dataNum)) if self.dataNum else None
        "allocate memory for change matrix of 0s"
        self.inputChange = np.zeros((self.hidden, self.input + 1)) if self.hidden and self.input else None
        self.outputChange = np.zeros((self.output, self.hidden + 1)) if self.output and self.hidden else None
        "calculate epsilon for randomization"
        self.hiddenEpsilon = np.sqrt(6.0 / (self.input + self.hidden)) if self.input and self.hidden else None
        self.outputEpsilon = np.sqrt(6.0 / (self.input + self.output)) if self.input and self.output else None
        "allocate memory for randomized weights"
        self.inputWeights = np.random.rand(self.hidden, self.input + 1) * 2 * self.hiddenEpsilon - self.hiddenEpsilon if self.hidden and self.input else None
        self.outputWeights = np.random.rand(self.output, self.hidden + 1) * 2 * self.outputEpsilon - self.outputEpsilon if self.output and self.hidden else None
        
    def getFeatures(self, dataType = 'd'):
        if dataType == 'd':
            self.trainingData = list(map(featureExtractor.Digit, self.rawTrainingData))
            self.testData = list(map(featureExtractor.Digit, self.rawTestData))
        elif dataType == 'f':
            self.trainingData = list(map(featureExtractor.Face, self.rawTrainingData))
            self.testData = list(map(featureExtractor.Face, self.rawTestData))
        return True
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def dsigmoid(self, y):
        return y * (1.0 - y)

    def genTruthMatrix(self, trainLabels):
        truth = np.zeros((self.output, self.dataNum))
        for i in range(self.dataNum):
            label = trainLabels[i]
            if self.output == 1:
                truth[:, i] = label
            else:
                truth[label, i] = 1
        return truth
    
    # comment about forward prop
    def forwardProp(self, thetaVec):
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))
        hiddenZ = self.inputWeights.dot(self.inputActivation)
        self.hiddenActivation[:-1, :] = self.sigmoid(hiddenZ)
        outputZ = self.outputWeights.dot(self.hiddenActivation)
        self.outputActivation = self.sigmoid(outputZ)
        # loss update
        costMatrix = self.outputTruth * np.log(self.outputActivation) + (1 - self.outputTruth) * np.log(
            1 - self.outputActivation)
        regulations = (np.sum(self.outputWeights[:, :-1] ** 2) + np.sum(self.inputWeights[:, :-1] ** 2)) * self.l / 2
        return (-costMatrix.sum() + regulations) / self.dataNum

    #comment about backwards prop
    def backwardProp(self, thetaVec):
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))
        outputError = self.outputActivation - self.outputTruth
        hiddenError = self.outputWeights[:, :-1].T.dot(outputError) * self.dsigmoid(self.hiddenActivation[:-1:])
        self.outputChange = outputError.dot(self.hiddenActivation.T) / self.dataNum
        self.inputChange = hiddenError.dot(self.inputActivation.T) / self.dataNum
        self.outputChange[:, :-1].__add__(self.l * self.outputWeights[:, :-1])
        self.inputChange[:, :-1].__add__(self.l * self.inputWeights[:, :-1])
        return np.append(self.inputChange.ravel(), self.outputChange.ravel())
    
    # training
    def train(self, iterations=3):
        for iteration in range(iterations):
            for index in range(len(self.trainingData)):
                y = self.trainingLabels[index]
                pred = self.classifier([self.trainingData[index]])[0]
                
                if pred != y:
                    self.weights[pred] -= self.trainingData[index]
                    self.weights[y] += self.trainingData[index]
                    
    def classify(self, testData):
        self.size_test = len(list(testData))
        features_test = []
        for datum in testData:
            feature = list(datum.values())
            features_test.append(feature)
        test_set = np.array(features_test, np.int32)
        feature_test_set = test_set.transpose()
        if feature_test_set.shape[1] != self.inputActivation.shape[1]:
            self.inputActivation = np.ones((self.input + 1, feature_test_set.shape[1]))
            self.hiddenActivation = np.ones((self.hidden + 1, feature_test_set.shape[1]))
            self.outputActivation = np.ones((self.output + 1, feature_test_set.shape[1]))
        self.inputActivation[:-1, :] = feature_test_set
        hiddenZ = self.inputWeights.dot(self.inputActivation)
        self.hiddenActivation[:-1, :] = self.sigmoid(hiddenZ)
        outputZ = self.outputWeights.dot(self.hiddenActivation)
        self.outputActivation = self.sigmoid(outputZ)
        if self.output > 1:
            return np.argmax(self.outputActivation, axis=0).tolist()
        else:
            return (self.outputActivation > 0.5).ravel()   
    
    def classifier(self, data):
        preds = []
        for datum in data:
            cols = util.Counter()
            for label in self.legalLabels:
                cols[label] = self.weights[label] * datum
            preds.append(cols.argMax())
        return preds
                    
    def test(self):
        self.predictions = self.classifier(self.testData)
        return self.predictions
    
    def run(self, iters = 5, debug=False):
        # adding this line fixed the code
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
                    print(f'Neural Network training with {dataType} data and size {trainingSize} [{int(size*100)}%] and iteration {index}...')
                    testStart = time.time()
                    self.train()
                    testTime = time.time() -  testStart
                    avgTime.append(testTime)
                    print(f'Neural Network testing with {dataType} data and size {trainingSize} [{int(size*100)}%] and iteration {index}...')
                    preds = self.test()
                    acc.append([preds[i] == self.testLabels[i] for i in range(len(self.testLabels))].count(True) / len(self.testLabels))
                    print(f'Neural Network prediction accuracy with {dataType} data and size {trainingSize} [{int(size*100)}%]: {acc[index] * 100}% (iteration {index})')
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
    classifierOne = NeuralNetworkClass()
    classifierOne.run(debug=True)
    classifierOne.write()