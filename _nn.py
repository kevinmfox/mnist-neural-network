import numpy as np
import pickle
from enum import Enum

class WeightInitMethod(Enum):
    Random = 1
    XavierUniform = 2
    XavierNormal = 3
    HeUniform = 4
    HeNormal = 5

class BiasInitMethod(Enum):
    Zero = 1
    Constant = 2
    Random = 3

class ActivationFunction(Enum):
    ReLU = 1
    Linear = 2
    Sigmoid = 3
    Softmax = 4
    LeakyReLU = 5
    ELU = 6
    TanH = 7

_logger = None

def setLogger(logger):
    global _logger  # pylint: disable=global-statement
    _logger = logger

def initializeNetwork(hiddenLayerDimensions, weightInitMethod=WeightInitMethod.Random, biasInitMethod=BiasInitMethod.Zero, randomSeed=None):
    # set the random seed
    np.random.seed(randomSeed)
    
    # define our layers
    inputLayerSize = 784 # the images are 28 x 28 (784)
    outputLayerSize = 10 # class evaluation from integers 0 - 9
    layers = [inputLayerSize] + hiddenLayerDimensions + [outputLayerSize]
    _logger.info(f'Initializing network parameters with dimensions: {layers}')
    _logger.info(f'Weight init method: {weightInitMethod.name}')
    _logger.info(f'Bias init method: {biasInitMethod.name}')

    parameters = {}
    for i in range(1, len(layers)):
        # initialize weights
        if weightInitMethod == weightInitMethod.Random:
            parameters[f'W{i}'] = np.random.randn(layers[i], layers[i-1]) * 0.01
        elif weightInitMethod == weightInitMethod.XavierUniform:
            limit = np.sqrt(6 / layers[i] + layers[i-1])
            parameters[f'W{i}'] = np.random.uniform(-limit, limit, (layers[i], layers[i-1]))
        elif weightInitMethod == weightInitMethod.XavierNormal:
            stddev = np.sqrt(2 / (layers[i] + layers[i-1]))
            parameters[f'W{i}'] = np.random.randn(layers[i], layers[i-1]) * stddev
        elif weightInitMethod == weightInitMethod.HeUniform:
            limit = np.sqrt(6 / layers[i-1])
            parameters[f'W{i}'] = np.random.uniform(-limit, limit, (layers[i], layers[i-1]))
        elif weightInitMethod == weightInitMethod.HeNormal:
            stddev = np.sqrt(2 / layers[i-1])
            parameters[f'W{i}'] = np.random.randn(layers[i], layers[i-1]) * stddev

        # initialize biases
        if biasInitMethod == biasInitMethod.Zero:
            parameters[f'b{i}'] = np.zeros((layers[i], 1))
        elif biasInitMethod == biasInitMethod.Constant:
            parameters[f'b{i}'] = np.full((layers[i], 1), 0.1)
        elif biasInitMethod == biasInitMethod.Random:
            stddev = np.sqrt(2 / layers[i-1])
            parameters[f'b{i}'] = np.random.randn(layers[i], 1) * stddev

    return parameters

def _forwardPropagation(X, parameters, hiddenActivation=ActivationFunction.ReLU, dropoutRate=None):
    def forward_relu(x):
        return np.maximum(0, x)

    def forward_linear(x):
        return x

    def forward_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward_softmax(x):
        exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)

    def forward_leakyRelu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def forward_elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def forward_tanh(x):
        return np.tanh(x)

    def forward_dropout(A, dropoutRate):
        mask = np.random.rand(*A.shape) > dropoutRate
        A_dropout = A * mask
        A_dropout /= (1 - dropoutRate)
        return A_dropout, mask

    A = X
    cache = {}
    layers = len(parameters) // 2
    dropoutMasks = []

    cache['A0'] = X

    for l in range(1, layers + 1):
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = np.dot(W, A) + b
        # we're in a hidden layer
        if l < layers:
            A = forward_relu(Z)
            if dropoutRate:
                A, mask = forward_dropout(A, dropoutRate)
                dropoutMasks.append(mask)
        # last layer
        else:
            A = forward_softmax(Z)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A

    return A, cache, dropoutMasks

def trainNetwork(
        parameters, 
        trainingData, 
        trainingLabels,
        validationData,
        validationLabels,
        hiddenActivation,
        learningRate,
        momentumBeta,
        rmsPropBeta,
        dropoutRate,
        optimizer,
        miniBatchSize,
        epochs,
        earlyStoppingPatience,
        earlyStoppingDelta,
        decayType,
        decayParam,
        randomSeed=None):

    def printStats(epoch, cost, accuracy):
        if accuracy is None:
            _logger.info(f'Epoch {epoch}: {cost}')
        else:
            _logger.info(f'Epoch {epoch}: Cost = {cost:.4f}; Accuracy = {accuracy:.2%}')

    def createBatches(X, Y, batchSize):
        samples = X.shape[1]
        miniBatches = []
        batchCount = samples // batchSize

        for i in range(batchCount):
            X_batch = X[:, i * batchSize:(i + 1) * batchSize]
            Y_batch = Y[:, i * batchSize:(i + 1) * batchSize]
            miniBatches.append((X_batch, Y_batch))

        if samples % batchSize != 0:
            X_batch = X[:, batchCount * batchSize:]
            Y_batch = Y[:, batchCount * batchSize:]
            miniBatches.append((X_batch, Y_batch))            

        return miniBatches

    def initializeMomentum(parameters):
        v = {}
        layers = len(parameters) // 2
        for l in range(1, layers + 1):
            v[f'dW{l}'] = np.zeros_like(parameters[f'W{l}'])
            v[f'db{l}'] = np.zeros_like(parameters[f'b{l}'])
        return v

    def initializeRmsProp(parameters):
        s = {}
        layers = len(parameters) // 2
        for l in range(1, layers + 1):
            s[f'dW{l}'] = np.zeros_like(parameters[f'W{l}'])
            s[f'db{l}'] = np.zeros_like(parameters[f'b{l}'])
        return s

    def backwardPropagation(X, Y, cache, parameters, hiddenActivation=ActivationFunction.ReLU, dropoutRate=None, dropoutMasks=None):
        def backward_relu(Z):
            return np.where(Z > 0, 1, 0)

        def backward_linear(Z):
            pass

        def backward_softmax(A, Y):
            return A - Y

        def backward_dropout(dA, mask, dropoutRate):
            dA *= mask
            dA /= (1 - dropoutRate)
            return dA

        samples = X.shape[1]
        layers = len(cache) // 2
        grads = {}

        for l in range(layers, 0, -1):
            A = cache[f'A{l}']
            A_prev = cache[f'A{l-1}']
            # last layer (softmax)
            if l == layers:
                dZ = backward_softmax(A, Y)
            # hidden layer (relu)
            else:
                W_next = parameters[f'W{l+1}']
                dA = np.dot(W_next.T, dZ)
                Z = cache[f'Z{l}']
                dZ = backward_relu(Z) * dA
                if dropoutRate:
                    dA = backward_dropout(dA, dropoutMasks[l-1], dropoutRate)
            grads[f'dW{l}'] = np.dot(dZ, A_prev.T) / samples
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / samples
        
        return grads

    def updateParameters(parameters, grads, momentumParams, rmsPropParams, learningRate, momentumBeta, rmsPropBeta, batchNumber):
        layers = len(parameters) // 2
        v_corrected = {}
        s_corrected = {}
        epsilon = 1e-8

        for l in range(1, layers + 1):
            # using momentum
            if momentumParams is not None and rmsPropParams is None:
                v = momentumParams
                v[f'dW{l}'] = momentumBeta * v[f'dW{l}'] + (1 - momentumBeta) * grads[f'dW{l}']
                v[f'db{l}'] = momentumBeta * v[f'db{l}'] + (1 - momentumBeta) * grads[f'db{l}']
                parameters[f'W{l}'] -= learningRate * v[f'dW{l}']
                parameters[f'b{l}'] -= learningRate * v[f'db{l}']
                momentumParams = v
            # using rmsprop
            elif rmsPropParams is not None and momentumParams is None:
                s = rmsPropParams
                s[f'dW{l}'] = rmsPropBeta * s[f'dW{l}'] + (1 - rmsPropBeta) * np.square(grads[f'dW{l}'])
                s[f'db{l}'] = rmsPropBeta * s[f'db{l}'] + (1 - rmsPropBeta) * np.square(grads[f'db{l}'])
                parameters[f'W{l}'] -= learningRate * grads[f'dW{l}'] / (np.sqrt(s[f'dW{l}'] + epsilon))
                parameters[f'b{l}'] -= learningRate * grads[f'db{l}'] / (np.sqrt(s[f'db{l}'] + epsilon))
                rmsPropParams = s
            # using adam
            elif momentumParams is not None and rmsPropParams is not None:
                v = momentumParams
                s = rmsPropParams
                v[f'dW{l}'] = momentumBeta * v[f'dW{l}'] + (1 - momentumBeta) * grads[f'dW{l}']
                v[f'db{l}'] = momentumBeta * v[f'db{l}'] + (1 - momentumBeta) * grads[f'db{l}']
                s[f'dW{l}'] = rmsPropBeta * s[f'dW{l}'] + (1 - rmsPropBeta) * np.square(grads[f'dW{l}'])
                s[f'db{l}'] = rmsPropBeta * s[f'db{l}'] + (1 - rmsPropBeta) * np.square(grads[f'db{l}'])
                v_corrected[f'dW{l}'] = v[f'dW{l}'] / (1 - momentumBeta ** batchNumber)
                v_corrected[f'db{l}'] = v[f'db{l}'] / (1 - momentumBeta ** batchNumber)
                s_corrected[f'dW{l}'] = s[f'dW{l}'] / (1 - rmsPropBeta ** batchNumber)
                s_corrected[f'db{l}'] = s[f'db{l}'] / (1 - rmsPropBeta ** batchNumber)
                parameters[f'W{l}'] -= learningRate * v_corrected[f'dW{l}'] / (np.sqrt(s_corrected[f'dW{l}'] + epsilon))
                parameters[f'b{l}'] -= learningRate * v_corrected[f'db{l}'] / (np.sqrt(s_corrected[f'db{l}'] + epsilon))
                momentumParams = v
                rmsPropParams = s
            # using gd
            else:
                parameters[f'W{l}'] -= learningRate * grads[f'dW{l}']
                parameters[f'b{l}'] -= learningRate * grads[f'db{l}']
    
        return parameters, momentumParams, rmsPropParams

    def updateLearningRate(decayType, decayParam, learningRate, epoch):
        if decayType == 'time':
            return learningRate / (1 + decayParam * epoch)
        elif decayType == 'exp':
            return learningRate * np.exp(-decayParam * epoch)

    # was used for troubleshooting
    def showShapes(X, Y, parameters, cache, grads):
        layers = len(parameters) // 2
        print(f'Layers: {layers}')
        print(f'X: {X.shape}')
        print(f'Y: {Y.shape}')
        for l in range(1, layers + 1):
            print(f"W{l}: {parameters[f'W{l}'].shape}")
            print(f"b{l}: {parameters[f'b{l}'].shape}")
        print(f"A0: {cache['A0'].shape}")
        for l in range(1, layers + 1):
            print(f"A{l}: {cache[f'A{l}'].shape}")
            print(f"Z{l}: {cache[f'Z{l}'].shape}")
        for l in range(1, layers + 1):
            print(f"dW{l}: {grads[f'dW{l}'].shape}")
            print(f"db{l}: {grads[f'db{l}'].shape}")

    np.random.seed(randomSeed)

    if dropoutRate == 0:
        dropoutRate = None
    if miniBatchSize >= trainingLabels.shape[1]:
        _logger.warning('Mini batch size is too big - will not use mini batch.')
        miniBatchSize = None
    if miniBatchSize == 0:
        miniBatchSize = None
    if optimizer in {'gd', 'momentum'}:
        rmsPropBeta = None
    if optimizer in {'gd', 'rmsprop'}:
        momentumBeta = None

    _logger.info('Training neural network')
    _logger.info(f'Hidden activation function: {hiddenActivation.name}')
    _logger.info(f'Optimizer: {optimizer}')
    _logger.info(f'Learning rate: {learningRate}')
    momentumBeta and _logger.info(f'Momentum beta: {momentumBeta}')
    rmsPropBeta and _logger.info(f'RMS Prop beta: {rmsPropBeta}')
    _logger.info(f'Mini-batch size: {miniBatchSize}')
    earlyStoppingPatience and _logger.info(f'Early stopping patience: {earlyStoppingPatience}')
    earlyStoppingDelta and _logger.info(f'Early stopping minimum delta: {earlyStoppingDelta}')
    decayType and _logger.info(f'Decay type: {decayType}')
    decayType and _logger.info(f'Decay param: {decayParam}')
    _logger.info(f'Epochs: {epochs}')

    costHistory = []
    accuracy = None
    accuracyHistory = []
    samples = trainingLabels.shape[1]
    batchNumber = 0
    patienceCounter = 0
    bestValidationCost = float('inf')
    earlyStop = False
    bestParameters = None

    # initialize some other parameters if we're using something other than 'gd' optimization
    momentumParams = initializeMomentum(parameters) if optimizer in {'momentum', 'adam'} else None
    rmsPropParams = initializeRmsProp(parameters) if optimizer in {'rmsprop', 'adam'} else None

    try:
        for epoch in range(1, epochs + 1):
            # check if we're using mini-batches
            if miniBatchSize is not None:
                # shuffle the data
                permutation = np.random.permutation(samples)
                trainingData = trainingData[:, permutation]
                trainingLabels = trainingLabels[:, permutation]
                # create the batches
                batches = createBatches(trainingData, trainingLabels, miniBatchSize)
            # not using mini-batches - create a single batch
            else:
                # create a single batch
                batches = []
                batches.append((trainingData, trainingLabels))
            
            # loop through the batches (may only be one batch if mini-batches are not used)
            batchCosts = []
            for X, Y in batches:
                batchNumber += 1
                batchSize = X.shape[1]

                # forward propagation
                A, cache, dropoutMasks = _forwardPropagation(X, parameters, hiddenActivation, dropoutRate)

                # backward propagation
                grads = backwardPropagation(X, Y, cache, parameters, hiddenActivation, dropoutRate, dropoutMasks)

                # update the parameters
                parameters, momentumParams, rmsPropParams = updateParameters(parameters, grads, momentumParams, rmsPropParams, learningRate, momentumBeta, rmsPropBeta, batchNumber)

                # calculate the batch cost
                batchCost = -np.sum(Y * np.log(A + 1e-9)) / batchSize
                batchCosts.append(batchCost)

            # calculate epoch cost
            cost = np.mean(batchCosts)
            costHistory.append(cost)
            batchCosts = []

            # check if we're adjusting our learning rate
            if decayType is not None:
                learningRate = updateLearningRate(decayType, decayParam, learningRate, epoch)

            # check if we're validation testing
            if validationData is not None:
                samples_val = validationData.shape[1]
                A_val, cache_val, _ = _forwardPropagation(validationData, parameters, hiddenActivation)
                cost_val = -np.sum(validationLabels * np.log(A_val + 1e-9)) / samples_val
                predictions = np.argmax(A_val, axis=0)
                true_labels = np.argmax(validationLabels, axis=0)
                accuracy = np.mean(predictions == true_labels)
                accuracyHistory.append(accuracy)
                # are we early stopping
                if earlyStoppingPatience is not None:
                    # we have a new best; reset patience
                    if cost_val < bestValidationCost - earlyStoppingDelta:
                        bestValidationCost = cost_val
                        patienceCounter = 0
                        bestParameters = parameters
                    # increment the patience counter
                    else:
                        patienceCounter += 1
                    earlyStop = patienceCounter >= earlyStoppingPatience

            if epoch % 100 == 0:
                printStats(epoch, cost, accuracy)
            
            # check if early stop has been triggered
            if earlyStop:
                _logger.important(f'Early stop triggered: {epoch}')
                parameters = bestParameters
                break
    except KeyboardInterrupt:
        _logger.warning('Interrupt detected')

    printStats(epoch, cost, accuracy)

    _logger.info(f'Training complete')
    return parameters, costHistory, accuracyHistory if len(accuracyHistory) > 0 else None

def testNetwork(testingData, testingLabels, parameters):
    _logger.info('Testing network')
    A_test, _, _ = _forwardPropagation(testingData, parameters)

    # calculate cost
    samples = testingLabels.shape[1]
    cost_test = -np.sum(testingLabels * np.log(A_test + 1e-9)) / samples
    
    # calculate accuracy
    predictions = np.argmax(A_test, axis=0)
    true_labels = np.argmax(testingLabels, axis=0)
    accuracy = np.mean(predictions == true_labels)

    _logger.info(f'Test results: Cost = {cost_test:.4f}; Accuracy = {accuracy:.2%}')
    _logger.info(f'Testing complete')

    return cost_test, accuracy

def testImage(testData, parameters, metadata):
    hiddenActivation = ActivationFunction.ReLU
    lastActivation = ActivationFunction.Softmax
    if metadata is not None:
        hiddenActivation = metadata['hiddenActivation']
    A, _, _ = _forwardPropagation(testData, parameters, hiddenActivation=hiddenActivation)
    return A