from . import constants, samples

def fetch(datatype, numTraining):
    if datatype=='d':
        numTest = constants.DIGITS_TEST_DATA_SIZE
        legalLabels = range(10)

        rawTrainingData, chosenList = samples.loadDataFile(
            "data/digitdata/trainingimages", 
            numTraining, 
            constants.DIGIT_DATUM_WIDTH,
            constants.DIGIT_DATUM_HEIGHT,
            True
        )
    
        trainingLabels = samples.loadLabelsFile(
            "data/digitdata/traininglabels", 
            chosenList
        )
        
        rawTestData, chosenList = samples.loadDataFile(
            "data/digitdata/testimages", 
            numTest, 
            constants.DIGIT_DATUM_WIDTH, 
            constants.DIGIT_DATUM_HEIGHT
        
        )

        testLabels = samples.loadLabelsFile(
            "data/digitdata/testlabels", 
            chosenList
        )

    elif datatype=='f':
        numTest = constants.FACE_TEST_DATA_SIZE
        legalLabels = range(2)

        rawTrainingData, chosenList = samples.loadDataFile(
            "data/facedata/facedatatrain", 
            numTraining,
            constants.FACE_DATUM_WIDTH,
            constants.FACE_DATUM_HEIGHT, 
            True
        )

        trainingLabels = samples.loadLabelsFile(
            "data/facedata/facedatatrainlabels",  
            chosenList
        )

        rawTestData, chosenList = samples.loadDataFile(
            "data/facedata/facedatatest", 
            numTest,
            constants.FACE_DATUM_WIDTH,
            constants.FACE_DATUM_HEIGHT
        )

        testLabels = samples.loadLabelsFile(
            "data/facedata/facedatatestlabels", 
            chosenList
        )
    else:
        return False

    return rawTrainingData, trainingLabels, rawTestData, testLabels, legalLabels