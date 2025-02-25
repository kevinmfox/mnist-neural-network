import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pygame
from tabulate import tabulate
import _nn
import _utils

# initialize logging
logger = _utils.initializeLogging()
logger.info('Starting program')
_nn.setLogger(logger)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', help='The random seed to be used.', type=int, default=None)
# training arguments
parser.add_argument('--train', help='Train the neural network against the test data', action='store_true')
parser.add_argument('--data_file', help='The file to use for training the network.', type=str, default='mnist_784.csv')
parser.add_argument('--hidden_dimensions', help='The dimensions of the hidden layers (e.g. \'[12, 24]\').', type=_utils.arg_int_list, default=[12])
parser.add_argument('--weight_init', help='The type of initialization for the weights.', choices=['Random', 'XavierUniform', 'XavierNormal', 'HeUniform', 'HeNormal'], default='Random')
parser.add_argument('--bias_init', help='The type of initialization for the biases.', choices=['Zero', 'Constant', 'Random'], default='Zero')
parser.add_argument('--hidden_activation', help='The type of activation for the hidden layers.', choices=['ReLU', 'Linear', 'Sigmoid', 'LeakyReLU', 'ELU', 'TanH'], default='ReLU')
parser.add_argument('--learning_rate', help='The learning rate of the network.', type=float, default=0.01)
parser.add_argument('--momentum_beta', help='Beta value to use when using Momentum optimizer.', type=float, default=0.9)
parser.add_argument('--rmsprop_beta', help='Beta value to use when using RMSProp optimizer.', type=float, default=0.9)
parser.add_argument('--adam_beta1', help='Beta1 value to use when using Adam optimizer.', type=float, default=0.9)
parser.add_argument('--adam_beta2', help='Beta2 value to use when using Adam optimizer.', type=float, default=0.999)
parser.add_argument('--epochs', help='The number of epochs/iterations to train the network.', type=int, default=1000)
parser.add_argument('--dropout_rate', help='The dropout rate for the network.', type=int, default=0)
parser.add_argument('--optimizer', help='The optimization algorithm to use.', choices=['gd', 'momentum', 'rmsprop', 'adam'], default='gd')
parser.add_argument('--batch_size', help='The size of mini-batches for training.', type=int, default=0)
parser.add_argument('--validation_size', help='The percent to reserve in our training data for validation. (0-50)', type=int, default=0)
parser.add_argument('--testing_size', help='The percent to reserve in our training data for testing. (0-50)', type=int, default=0)
parser.add_argument('--early_stopping', help='Use Early Stopping during training.', action='store_true')
parser.add_argument('--early_stopping_patience', help='Number of epochs to wait before stopping.', type=int, default=None)
parser.add_argument('--early_stopping_delta', help='The minimum delta to use when early stopping is enabled.', type=float, default=0)
parser.add_argument('--decay_type', help='The type of decay to apply to the learning rate.', choices=['time', 'exp'], default=None)
parser.add_argument('--decay_param', help='The parameter to be used by the decay_type.', type=float, default=None)
# other arguments
parser.add_argument('--read_model', help='Read the information from a saved pickle model file.', action='store_true')
parser.add_argument('--model_file', help='The name of the pickle file to use.', type=str, default='model.pkl')
parser.add_argument('--show_charts', action='store_true')
parser.add_argument('--show_image', help='Will show an image from the training set.', action='store_true')
parser.add_argument('--index', help='The index of the image to show.', type=int, default=0)
parser.add_argument('--no_model', help='Will allow the pygame interface to run without a model.', action='store_true')
# dataset arguments
parser.add_argument('--create_dataset', help='Will create a dataset from a source dataset.', action='store_true')
parser.add_argument('--save_file', help='The file to save the new dataset to.', type=str, default='subset.csv')
parser.add_argument('--samples', help='The number of random samples to save in the new dataset from the source.', type=int, default=1000)
args = parser.parse_args()

### PYGAME ###
def runPygame(modelParameters, modelMetadata):
    def printText(text, location, size = 16):
        font = pygame.font.SysFont('Courier', size)
        text = font.render(text, True, (255, 255, 255), (0, 0, 0))
        textRect = location
        screen.blit(text, textRect)

    def resetScreen():
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, drawingAreaBackgroundColor, drawingAreaRect)
        for i in range(10):
            printText(str(i), (i * 35 + 35, 560), 24)

    def drawProbabilities(probabilities):
        probabilities = np.squeeze(probabilities)
        bottom = 500
        pygame.draw.rect(screen, (0, 0, 0), (0, 450, screenWidth, 100))
        for i in range(10):
            height = probabilities[i] * 100
            top = 550 - height
            pygame.draw.rect(screen, probabilityRectColor, (i * 35 + 32, top, 20, height))

    def getImageData():
        # get the data to be assessed
        drawRect = pygame.Rect(drawingAreaRect)
        # check if we have anything to assess yet
        if drawBounds[2] <= drawBounds[0] or drawBounds[3] <= drawBounds[1]:
            return None
        buffer = 0.2
        left = drawBounds[0]
        top = drawBounds[1]
        right = drawBounds[2]
        bottom = drawBounds[3]
        width = right - left
        height = bottom - top
        if width > height * 1.2:
            top = top - ((width - height) / 2)
            bottom = bottom + ((width - height) / 2)
            height = bottom - top
        elif height > width * 1.2:
            left = left - ((height - width) / 2)
            right = right + ((height - width) / 2)
            width = right - left
        xOffset = width * buffer
        yOffset = height * buffer
        drawRect = pygame.Rect(
            max(drawBounds[0] - xOffset, drawingAreaRect[0]), 
            max(drawBounds[1] - yOffset, drawingAreaRect[1]),
            min((drawBounds[2] - drawBounds[0]) + (2 * xOffset), drawingAreaRect[2]),
            min((drawBounds[3] - drawBounds[1]) + (2 * yOffset), drawingAreaRect[3])
        )
        subsurface = screen.subsurface(drawRect)
        imageData = pygame.surfarray.array3d(subsurface)
        imageData = cv2.resize(imageData, (28, 28), interpolation=cv2.INTER_AREA)
        # reshape the data for the neural netwrk model
        imageData = np.mean(imageData, axis=-1)     # convert to grayscale
        imageData = np.flipud(imageData)            # flip the image vertically
        imageData = np.rot90(imageData, k=-1)       # rotate 90 degrees
        imageData = imageData.reshape(28 * 28, 1)   # reshape to 28 x 28
        imageData = 255 - imageData                 # flip to white on black (this is what the network was trained on)
        imageData = imageData / 255.                # normalize the values
        return imageData

    # if we don't have a model, create a random one
    if modelParameters is None:
        modelParameters = _nn.initializeNetwork([12])

    # initialize pygame
    pygame.init()
    screenWidth = 400
    screenHeight = 600
    screen = pygame.display.set_mode((screenWidth, screenHeight), pygame.DOUBLEBUF)
    pygame.display.set_caption('Neural Network - MNIST')

    # variables
    drawingAreaBackgroundColor = (255, 255, 255)
    drawingAreaForegroundColor = (0, 0, 0)
    drawingAreaRect = pygame.Rect(0, 0, screenWidth, 400)
    drawBounds = [drawingAreaRect.width, drawingAreaRect.height, 0, 0]
    probabilityRectColor = (0, 200, 0)

    resetScreen()

    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left mouse button
                    drawing = True
                elif event.button == 3:  # right mouse button
                    resetScreen()
                    drawBounds = [drawingAreaRect.width, drawingAreaRect.height, 0, 0]
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    x, y = event.pos
                    if drawingAreaRect.collidepoint(event.pos):
                        pygame.draw.circle(screen, drawingAreaForegroundColor, event.pos, 10)
                        drawBounds[0] = min(drawBounds[0], x)
                        drawBounds[1] = min(drawBounds[1], y)
                        drawBounds[2] = max(drawBounds[2], x)
                        drawBounds[3] = max(drawBounds[3], y)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    imageData = getImageData()
                    showImage(imageData, None, 0)

        pygame.display.update()

        # test the image
        imageData = getImageData()
        if imageData is not None:
            probabilities = _nn.testImage(imageData, modelParameters, modelMetadata)
            drawProbabilities(probabilities)

def loadDataFromCsv(file, validationSize, testingSize, randomSeed=None):
    np.random.seed(randomSeed)
    logger.info(f'Loading data from file: {file}')
    try:
        # read the file
        data = pd.read_csv(file)
        imageData = data.iloc[:, :-1].values
        labelData = data.iloc[:, -1].values
        
        # check the data
        if imageData.shape[1] != 784 or labelData.shape[0] != imageData.shape[0]:
            raise ValueError('Data does not match expected shape.')
        logger.info(f'Data loaded: {imageData.shape[0]}')
        
        # reshape and prepare the data
        logger.info('Preparing data')
        imageData = imageData.reshape(imageData.shape[0], -1).T
        imageData = imageData / 255.0
        labelData = np.eye(10)[labelData]
        labelData = labelData.reshape(labelData.shape[0], -1).T
        indices = np.random.permutation(imageData.shape[1])
        imageData = imageData[:, indices]
        labelData = labelData[:, indices]

        # distribute our data sets
        totalSamples = len(imageData[1])
        trainingData = imageData
        trainingLabels = labelData
        validationData = None
        validationLabels = None
        testingData = None
        testingLabels = None
        # check if we want a validation set
        if validationSize > 0:
            logger.info(f'Creating validation subset: {validationSize}%')
            count = max(int(totalSamples * (validationSize / 100.0)), 1)
            validationData = trainingData[:, :count]
            validationLabels = trainingLabels[:, :count]
            trainingData = trainingData[:, count:]
            trainingLabels = trainingLabels[:, count:]
        # check if we want a testing set
        if testingSize > 0:
            logger.info(f'Creating testing subset: {testingSize}%')
            count = max(int(totalSamples * (testingSize / 100.0)), 1)
            testingData = trainingData[:, :count]
            testingLabels = trainingLabels[:, :count]
            trainingData = trainingData[:, count:]
            trainingLabels = trainingLabels[:, count:]

        logger.info(f'Training samples: {trainingData.shape[1]}')
        if validationSize > 0:
            logger.info(f'Validation samples: {validationData.shape[1]}')
        if testingSize > 0:
            logger.info(f'Testing samples: {testingData.shape[1]}')
        
        return trainingData, trainingLabels, validationData, validationLabels, testingData, testingLabels
    except Exception as e:
        logger.error(f'Failed to read data: {e}')
        exit()

def showImage(imageData, labelData, index):
    imagePixels = imageData[:, index] * 255.0
    imageShape = (28, 28)
    image = imagePixels.reshape(imageShape)
    if labelData is not None:
        logger.info(f'Image label: {np.argmax(labelData[:, index])}')
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()

def showCostAndAccuracyChart(costHistory, accuracyHistory):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cost', color='blue')
    ax1.plot(np.arange(1, len(costHistory) + 1), costHistory, color='blue', label='Cost')
    ax1.tick_params(axis='y', labelcolor='blue')
    if accuracyHistory:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)', color='green')
        ax2.plot(np.arange(1, len(accuracyHistory) + 1), accuracyHistory, color='green', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor='green')
        plt.title('Cost and Accuracy')
    else:
        plt.title('Cost')
    plt.show()

# we want to train our network
if args.train:
    logger.info('Training the neural network (--train)')

    trainingFile = args.data_file
    hiddenDimensions = args.hidden_dimensions
    epochs = args.epochs
    randomSeed = args.random_seed
    weightInitialization = args.weight_init
    weightInitialization = getattr(_nn.WeightInitMethod, weightInitialization)
    biasInitialization = args.bias_init
    biasInitialization = getattr(_nn.BiasInitMethod, biasInitialization)
    hiddenActivation = args.hidden_activation
    hiddenActivation = getattr(_nn.ActivationFunction, hiddenActivation)
    learningRate = args.learning_rate
    momentumBeta = args.momentum_beta
    rmsPropBeta = args.rmsprop_beta
    adamBeta1 = args.adam_beta1
    adamBeta2 = args.adam_beta2
    dropoutRate = args.dropout_rate / 100.
    optimizer = args.optimizer
    miniBatchSize = args.batch_size
    earlyStopping = args.early_stopping
    earlyStoppingPatience = args.early_stopping_patience if earlyStopping else None
    earlyStoppingDelta = args.early_stopping_delta if earlyStopping else None
    decayType = args.decay_type
    decayParam = args.decay_param
    validationSize = args.validation_size
    testingSize = args.testing_size
    saveFile = args.model_file
    
    # move some variables around if we're using adam optimization
    if optimizer == 'adam':
        momentumBeta = adamBeta1
        rmsPropBeta = adamBeta2

    # some input validations
    if validationSize not in range(0, 51):
        logger.error('Invalid validation_size. Needs to be 0-50.')
        exit()
    if testingSize not in range(0, 51):
        logger.error('Invalid testing_size. Needs to be 0-50.')
        exit()
    if validationSize + testingSize >= 80:
        logger.error('Validation and testing subsets are too large. Needs to be less than 80.')
        exit()
    if dropoutRate >= 60:
        logger.error('Dropout rate is too high. Needs to be less than 60.')
        exit()
    if validationSize == 0 and earlyStopping is True:
        logger.error('Cannot use early stopping without a validation set.')
        exit()

    # adjust for early stopping
    if earlyStopping is True and earlyStoppingPatience is None:
        earlyStoppingPatience = int(epochs * 0.1)

    # adjust decay params if applicable
    decay_defaults = {'time': 0.01, 'exp': 0.01}
    if decayType and decayParam is None:
        decayParam = decay_defaults.get(decayType, decayParam)

    # show a message if a random seed was specified
    if randomSeed:
        logger.important(f'Random seed specified: {randomSeed}')

    # load the training data
    trainingData, trainingLabels, validationData, validationLabels, testingData, testingLabels = loadDataFromCsv(trainingFile, validationSize, testingSize, randomSeed=randomSeed)

    # initilize the network
    parameters = _nn.initializeNetwork(
        hiddenLayerDimensions=hiddenDimensions,
        weightInitMethod=weightInitialization,
        biasInitMethod=biasInitialization,
        randomSeed=randomSeed)

    # train the network
    parameters, costHistory, accuracyHistory = _nn.trainNetwork(
        parameters=parameters, 
        trainingData=trainingData,
        trainingLabels=trainingLabels,
        validationData=validationData,
        validationLabels=validationLabels,
        hiddenActivation=hiddenActivation,
        learningRate=learningRate,
        momentumBeta=momentumBeta,
        rmsPropBeta=rmsPropBeta,
        dropoutRate=dropoutRate,
        optimizer=optimizer,
        miniBatchSize=miniBatchSize,
        epochs=epochs,
        earlyStoppingPatience=earlyStoppingPatience,
        earlyStoppingDelta=earlyStoppingDelta,
        decayType=decayType,
        decayParam=decayParam,
        randomSeed=randomSeed)
    trainingCost = costHistory[-1]
    trainingAccuracy = accuracyHistory[-1] if accuracyHistory else None

    # check if we want to show some charts
    if args.show_charts:
        showCostAndAccuracyChart(costHistory, accuracyHistory)
    
    # test the network
    testingCost = None
    testingAccuracy = None
    if testingData is not None:
        testingCost, testingAccuracy = _nn.testNetwork(testingData, testingLabels, parameters)

    # check if we're saving the model to a file
    if saveFile:
        logger.info(f'Saving model to file: {saveFile}')
        metadata = {
            'hiddenLayerDimensions': hiddenDimensions,
            'weightInitMethod': weightInitialization.name,
            'biasInitMethod': biasInitialization.name,
            'randomSeed': randomSeed,
            'hiddenActivation': hiddenActivation.name,
            'learningRate': learningRate,
            'momentumBeta': momentumBeta,
            'rmsPropBeta': rmsPropBeta,
            'dropoutRate': dropoutRate,
            'optimizer': optimizer,
            'earlyStoppingPatience': earlyStoppingPatience,
            'earlyStoppingDelta': earlyStoppingDelta,
            'decayType': decayType,
            'decayParam': decayParam,
            'epochs': epochs,
            'trainingSize': trainingData.shape[1],
            'validationSize': validationData.shape[1] if validationData is not None else None,
            'testingSize': testingData.shape[1] if testingData is not None else None,
            'trainingCost': round(trainingCost, 4),
            'trainingAccuracy': round(trainingAccuracy, 4) if trainingAccuracy is not None else None,
            'testingCost': round(testingCost, 4) if testingCost is not None else None,
            'testingAccuracy': round(testingAccuracy, 4) if testingAccuracy is not None else None,
        }
        modelInfo = {
            'metadata': metadata,
            'parameters': parameters,
            'costHistory': costHistory,
            'accuracyHistory': accuracyHistory if not None else None,
        }
        try:
            with open(saveFile, 'wb') as f:
                pickle.dump(modelInfo, f)
        except Exception as e:
            logger.error(f'Failed to save model fle: {e}')

# reading a model file
elif args.read_model:
    modelFile = args.model_file
    logger.info(f'Reading model file: {modelFile}')
    try:
        with open(modelFile, 'rb') as f:
            modelInfo = pickle.load(f)
        metadata = modelInfo['metadata']
        parameters = modelInfo['parameters']
        metadata_list = [[key, value] for key, value in metadata.items()]
        print(tabulate(metadata_list, headers=['Parameter', 'Value'], tablefmt='pretty'))
        # check if we want to show charts
        if args.show_charts:
            costHistory = modelInfo['costHistory']
            accuracyHistory = modelInfo['accuracyHistory']
            showCostAndAccuracyChart(costHistory, accuracyHistory)
    except Exception as e:
        logger.error(f'Failed to read model file: {e}')
        exit()

# showing an image
elif args.show_image:
    trainingFile = args.data_file
    imageIndex = args.index

    # load the training data
    trainingData, trainingLabels, _, _, _, _ = loadDataFromCsv(trainingFile, 0, 0)

    # show an image
    showImage(trainingData, trainingLabels, imageIndex)

# creating a dataset
elif args.create_dataset:
    # grab some parameters
    sourceFile = args.data_file
    samples = args.samples
    randomSeed = args.random_seed
    saveFile = args.save_file

    # get the source data
    logger.info(f'Creating new dataset')
    logger.info(f'Source file: {sourceFile}')
    data = pd.read_csv(sourceFile)
    logger.info(f'Samples: {len(data)}')

    # save a subset
    randomSeed and logger.important(f'Random seed: {randomSeed}')
    if samples >= len(data):
        logger.error('Not enough samples in source file.')
    logger.info(f'Selecting samples: {samples}')
    subset = data.sample(n=samples, random_state=randomSeed)
    logger.info(f'Saving to file: {saveFile}')
    subset.to_csv(saveFile, index=False)

# running the pygame UI
else:
    logger.info('Running user interface via pygame')
    modelFile = args.model_file
    
    if not args.no_model:
        logger.info(f'Using model file: {modelFile}')
    
        # read the model file
        try:
            with open(modelFile, 'rb') as f:
                modelInfo = pickle.load(f)
            metadata = modelInfo['metadata']
            parameters = modelInfo['parameters']
        except Exception as e:
            logger.error(f'Failed to read model file: {e}')
            exit()
    else:
        logger.warning('Running with no model.')
        parameters = None
        metadata = None

    # run the 'game'
    runPygame(parameters, metadata)

logger.info('All done')
