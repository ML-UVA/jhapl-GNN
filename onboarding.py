import numpy as np
import random
from sknetwork.gnn import GNNClassifier

def make_mountain_graph():
    xAxis = np.linspace(0, 1, num = 100)
    yAxis = np.linspace(0, 1, num = 100)


    heights = np.zeros((100, 100))
    X, Y = np.meshgrid(xAxis, yAxis)

    for i in range(20): 
        A = random.randint(50, 100)
        x0 = np.random.rand()
        y0 = np.random.rand()
        heightValue = A * np.exp(-A * ((X - x0)**2 + (Y - y0)**2))
        heights += heightValue

    mean = 0 
    std_dev = 0.2 
    noise = np.random.normal(mean, std_dev, size=heights.shape)
    heights_noise = heights + noise

    numberOfNodes = 10000
    nodes = np.arange(numberOfNodes).reshape(100,100)
    rows =[]
    cols = []
    edgeWeights = []


    east = nodes[:, :-1].ravel()
    destination = nodes[:, 1:].ravel()
    weights = np.abs(heights_noise[:, 1:].ravel() - heights_noise[:, :-1].ravel())
    rows.append((east))
    cols.append(destination)
    edgeWeights.append(weights)

    west_src = nodes[:, 1:].ravel()
    west_dst = nodes[:, :-1].ravel()
    west_w  = np.abs(heights_noise[:, :-1].ravel() - heights_noise[:, 1:].ravel())
    rows.append(west_src)
    cols.append(west_dst)
    edgeWeights.append(west_w)

    south_src = nodes[:-1, :].ravel()
    south_dst = nodes[1:, :].ravel()
    south_w  = np.abs(heights_noise[1:, :].ravel() - heights_noise[:-1, :].ravel())
    rows.append(south_src)
    cols.append(south_dst)
    edgeWeights.append(south_w)

    north_src = nodes[1:, :].ravel()
    north_dst = nodes[:-1, :].ravel()
    north_w  = np.abs(heights_noise[:-1, :].ravel() - heights_noise[1:, :].ravel())

    rows.append(north_src)
    cols.append(north_dst)
    edgeWeights.append(north_w)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    edgeWeights = np.concatenate(edgeWeights)

    from scipy import sparse

    adjacency = sparse.csr_matrix(
        (edgeWeights, (rows, cols)),
        shape=(numberOfNodes, numberOfNodes)
    )

    heightFlat = heights_noise.ravel()
    noise = np.random.normal(0, 1.0, size=heightFlat.shape)
    humidity = 0.3 * heightFlat + noise  
    features = np.stack([heightFlat, humidity], axis=1) 

    up    = np.pad(heights_noise[1:, :],  ((0, 1), (0, 0)), constant_values=-1)
    down  = np.pad(heights_noise[:-1, :], ((1, 0), (0, 0)), constant_values=-1)
    left  = np.pad(heights_noise[:, 1:],  ((0, 0), (0, 1)), constant_values=-1)
    right = np.pad(heights_noise[:, :-1], ((0, 0), (1, 0)), constant_values=-1)

    greaterThanUp = heights > up
    greaterThanDown = heights > down
    greaterThanLeft = heights > left
    greaterThanRgiht = heights > right

    temp1 = greaterThanUp & greaterThanDown
    temp2 = greaterThanLeft & greaterThanRgiht
    peaks = temp1 & temp2

    labels = peaks.ravel().astype(int)
    return adjacency, features, labels
from scipy import sparse
train, featurestrain, trainlabel = make_mountain_graph()
test, featurestest, testlabel = make_mountain_graph()

testLabelsGone = -1 * np.ones_like(testlabel)
allFeatures = np.concatenate((featurestrain, featurestest), axis=0)
allLabels = np.append(trainlabel, testLabelsGone)
allAdjacency = sparse.block_diag([train, test], format="csr")

hyperparameters = [{"dims": [8, 2], "lr": 0.01}, {"dims": [16, 2], "lr": 0.01}, {"dims": [32, 2], "lr": 0.01}, {"dims": [16, 16, 2], "lr": 0.01}, {"dims": [16, 2], "lr": 0.005}]

for i in hyperparameters:
    gnn = GNNClassifier(dims=i["dims"],optimizer='Adam', learning_rate=i["lr"], early_stopping=True)
    gnn.fit(allAdjacency, allFeatures, allLabels)
    allPred = gnn.predict()

    num_train_nodes = trainlabel.shape[0]
    trainingPredictoin = allPred[:num_train_nodes]
    testprediction = allPred[num_train_nodes:]
    train_acc = (trainingPredictoin == trainlabel).mean()
    test_acc = (testprediction == testlabel).mean()

    print("dims =", i["dims"], "lr =", i["lr"], "\n train_acc =", (train_acc), "test_acc =", (test_acc))
