import numpy as np
import pandas as pd
import ctypes
import matplotlib.pyplot as plt

# Note:
# 1. testing is not implemented yet. only training is implemented. - Done
# 2. implement accuracy and other metrics. - Done
# 3. implement opencl and cuda backend for layers. - rewrite .. they keep failing at either extreme values or large datasets.
# 4. add save and reuse functionality.
# 5. maybe add a simple preprocessor extend functionality. 1/2
# 6. maybe code a simple parameter randomizer for testing and experimentation purposes.
# 7. timers for experimentation purposes. maybe make a simple class for this that can be used as a context manager.


class Network:
    def __init__(self, layers:list)->None:
        # layers should be a list of Layers objects.
        # The first layer should be the input layer and the last layer should be the output layer. 
        # The input layer is not used for calculations but is used to set the input shape for the first hidden layer. 
        # The output layer is used to set the output shape for the last hidden layer. 
        # The hidden & output layers are used for calculations and can have any activation function and weight distribution.
        self.layers = layers

    # Compile should be called after initializing the network and before training. 
    # It sets the loss function, metric functions, and generates weights and biases for all layers except the input layer. 
    # It also sets the learning rate for all layers.
    def Compile(self, LearningRate:float=0.01, Loss:str="CCE", metrics:list=["Accuracy"], weightDist:str="default", force:bool=False) -> None:
        # checks for the last layer activation slope and sets the loss function accordingly. 
        # If the last layer activation slope is a string, it assumes its a loss function and sets it as the loss function. 
        # If its not a string, it assumes its a function and sets the loss function to the one provided in the Loss parameter. 
        # If force is True, it will set the loss function to the one provided in the Loss parameter regardless of the last layer activation slope.
        lastLayerSimplify = self.layers[-1].activation_slope
        if not callable(lastLayerSimplify) and not force:
            Loss = lastLayerSimplify
            print(f"Auto-set loss to {Loss}. Set force=True to override.")

        elif force:
            print(f"Warning: Loss function {Loss} is forced. Its not an error but please confirm {Loss} is appropriate for last layer slope")

        else:
            print(f"Warning: last layer is a function. Its not an error but please confirm {Loss} is appropriate")

        if "F1" in metrics and ("Precision" in metrics or "Recall" in metrics):
            print("Warning: F1 is set. Precision and Recall will be ignored in metric calculations to avoid redundancy. setting force=True does not override this.")

        # set the loss function and metric functions based on the provided keys.
        loss_functs = {
            "CCE" : self.catCrossEnt,
            "BCE" : self.binCrossEnt
        }
        metric_functs = {
            "Accuracy" : self.accuracy,
            "Precision" : self.precision,
            "Recall" : self.recall,
            "F1" : self.f1
        }

        try:
            self.loss = loss_functs[Loss]

        except KeyError:
            print(f"KeyError: {Loss} is not a valid loss Key.\nValid keys are {loss_functs.keys()}.\n")
            return

        except Exception as e:
            print(f"{e}:HUH !!!")
            return

        try:
            self.metrics = {}
            

            for x in metrics:
                self.metrics[x] = metric_functs[x]

        except KeyError:
            print(f"KeyError: {x} is not a valid metric Key.\nValid keys are {metric_functs.keys()}.\n")
            return

        except Exception as e:
            print(f"{e}:HUH !!!")
            return

        # generate weights and biases for all layers except the input layer. set learning rate for all layers.
        init_ocl = False
        init_cuda = False
        for i in range(1, len(self.layers)):
            
            self.layers[i].genWeightsBiases(self.layers[i-1].neuronLength, weightDist=weightDist)
            self.layers[i].lr = LearningRate

            if self.layers[i].backend == "openCl":
                if not init_ocl:
                    self.initOpenCl()
                    init_ocl = True
                self.layers[i].borrow = self.lib.snn_forward

                
    # Forward should be called before Backward. It takes the input data and passes it through the network to get the output.
    def Forward(self, train:np.ndarray) -> None:
        # the input data is passed through the network layer by layer. 
        # The output of each layer is stored in the layer object for use in the backward pass. 
        # The final output is stored in the network object for use in the loss function and metric functions.

        inp = train # contains the training data. it is passed through the network layer by layer.

        # simple forward pass. the output of each layer is stored in the layer object for use in the backward pass.
        for layer in self.layers[1:]:
            inp = layer.forward(inp)

        self.output = inp # stores the final output of the network for use in the loss function and metric functions.
        
    # Backward should be called after Forward. It takes the output of the network and the target values and calculates the gradients for each layer and updates the weights and biases accordingly.
    def Backward(self):
        dL_dn = self.layers[-1].backward(self.dL_do) # the gradient of the loss with respect to the output of the network. it is calculated in the loss function and stored in the network object for use in the backward pass.

        # simple backward pass. the gradient of the loss with respect to the output of the network is passed through the network layer by layer in reverse order.
        for layer in reversed(self.layers[1:-1]):
            dL_dn = layer.backward(dL_dn)

    # Fit should be called after Compile. 
    # It takes the training data and the number of epochs and trains the network by calling Forward and Backward for each epoch. 
    # It also stores the loss for each epoch in the network object for use in plotting the loss curve.
    def Fit(self, train, epochs=100, n=10, batch=1000):
        #targets = self.encode(train[1]) fix this later. only encode if requested and appropriate parameters are set. for now, just assume the targets are already encoded.
        self.pLoss = [] # stores loss for each epoch.
        targets = train[1] # change this later. only encode if requested and appropriate parameters are set. for now, just assume the targets are already encoded.
        self.sMV = {}
        batch = batch
        for metric in self.metrics:
            self.sMV[metric] = [] # stores values for each epoch.
        # simple training loop.
        for epoch in range(epochs):
            for i in range(0, len(train[0]), batch):
                self.Forward(train[0][i:i+batch])
                self.dL_do = self.loss(targets[i:i+batch]) # precariously stores the gradient of the loss with respect to the output of the network for use in the backward pass.
                self.Backward()
                self.pLoss.append(float(self.Loss))
                # if epoch % (epochs // n) == 0: print(f"Epoch {epoch}/{epochs} - Batch {i//batch + 1}/{len(train[0])//batch} - Loss: {self.Loss:.4f}")

                self.calcMetrics(targets[i:i+batch], self.sMV)

            # some kinda progress loader

            # for i in range(1,10):
            #     if epoch % (epochs // n) == i:
            #         print("_",end='\r')

            if epoch % (epochs // n) == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {self.Loss}")

    # the loss functions return the gradient of the loss with respect to the output of the network and also store the loss in the network object for use in plotting the loss curve.
    def catCrossEnt(self, target):
        x = self.output - target
        self.Loss = -np.mean(np.sum(target * np.log(np.clip(self.output, 1e-7, 1)), axis=1))
        return x

    def binCrossEnt(self, target):
        x = self.output - target
        self.Loss = -np.mean(target * np.log(np.clip(self.output, 1e-7, 1)) + (1 - target) * np.log(np.clip(1 - self.output, 1e-7, 1)))
        return x
    
    def Test(self, test):
        targets = test[1] 
        self.Forward(test[0])
        self.loss(targets)
        self.testMetrics = {}
        for metric in self.metrics:
            self.testMetrics[metric] = []
        self.calcMetrics(targets, self.testMetrics)
        print(f"Test Loss: {self.Loss}")
        for metric, values in self.testMetrics.items():
            print(f"Test {metric}: {np.mean(values)}")

    def accuracy(self, target):
        predicted = np.argmax(self.output, axis=1)
        actual = np.argmax(target, axis=1)
        return np.mean(predicted == actual)
    
    def precision(self, target):
        predicted = np.argmax(self.output, axis=1)
        actual = np.argmax(target, axis=1)
        tp = np.sum((predicted == 1) & (actual == 1))
        fp = np.sum((predicted == 1) & (actual == 0))
        return tp / (tp + fp + 1e-7)
    
    def recall(self, target):
        predicted = np.argmax(self.output, axis=1)
        actual = np.argmax(target, axis=1)
        tp = np.sum((predicted == 1) & (actual == 1))
        fn = np.sum((predicted == 0) & (actual == 1))
        return tp / (tp + fn + 1e-7)

    def f1(self, target):
        precision = self.precision(target)
        recall = self.recall(target)
        return (2 * (precision * recall) / (precision + recall + 1e-7), precision, recall)
    
    def calcMetrics(self, target, store):
        for metric in self.metrics:
                if metric == "F1":
                    f1, precision, recall = self.metrics[metric](target)
                    store[metric].append(float(f1))
                    if "Precision" in self.metrics:
                        store["Precision"].append(float(precision))
                    if "Recall" in self.metrics:
                        store["Recall"].append(float(recall))

                elif (metric == "Precision" or metric == "Recall") and "F1" in self.metrics:
                    continue

                else:
                    store[metric].append(float(self.metrics[metric](target)))
    
    def plotMetrics(self):
        x = np.arange(start=0, step=1, stop= len(self.pLoss))
        y = self.pLoss
        plt.plot(x, y, label="Loss")
        for metric in self.metrics:
            plt.plot(x, self.sMV[metric], label=metric)
        plt.legend()
        plt.show()

    def initOpenCl(self):
        self.lib = ctypes.CDLL("./D-ops/main.dll")
        self.lib.init_opencl()
        self.lib.init_snn()

        self.lib.snn_forward.argtypes  = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]

    
class Preprocessor():
    def __init__(self, train:tuple, test:tuple)->None:
        self.train_in = train[0]
        self.train_out = train[1]
        self.test_in = test[0]
        self.test_out = test[1]
        pass

    def labels(self, into:str="onehot"):
        if into == "onehot":
            trainLabels = np.unique(self.train_out)
            testLabels = np.unique(self.test_out)
            if not np.array_equal(trainLabels, testLabels):
                raise ValueError("Warning: Training and test labels are not equal.")
            
            else:
                print("Training and test labels are equal. Proceeding with one-hot encoding.")
                self.train_out = np.array([[1 if label == x else 0 for x in trainLabels] for label in self.train_out])
                self.test_out = np.array([[1 if label == x else 0 for x in testLabels] for label in self.test_out])
                print("One-hot encoding complete.")
    
    def features(self, into:str="normalize"):
        if into == "normalize":
            train_max = np.max(self.train_in)
            test_max = np.max(self.test_in)

            
            self.train_in = self.train_in / train_max
            self.test_in = self.test_in / test_max
            print("Normalization complete.")

    def data(self):
        return (self.train_in, self.train_out), (self.test_in, self.test_out)

# Layers class represents a layer in the neural network. It contains the weights, biases, activation function, and other parameters for the layer.
class Layers:
    # layer initializer. size is mandatory. activation, alpha, and backend are optional.
    # input layers dont really need anything else.
    # output layer is just a hidden layer with a specific activation function and weight distribution. must be set to accomodate network.compile() parameters.
    def __init__(self, size:int, activation:str="relu", alpha:float=0.1, backend:str="numpy"):
        # the activation functions and matmul functions are stored in dictionaries for easy access based on the provided keys.
        act_functs = {
            "relu" : [self.Relu, self.ReluSlope],
            "lrelu" : [self.LRelu, self.LReluSlope],
            "sigmoid" : [self.sigmoid, "CCE"],
            "SFMX" : [self.SoftMax, "BCE"]
        }
        matmul_functs = {
            "numpy": self.numpyMatmul, 
            "openCl": self.openCl
            }
        
        # size is the number of neurons in the layer. activation is the activation function for the layer. 
        # alpha is the slope for leaky relu. 
        # backend is the backend for matrix multiplication.
        self.neuronLength = size
        self.alpha = alpha
        self.borrow = None
        try:
            if activation == "LRelu":
                print(f"Warning: alpha already set. Set layers(alpha) at initialization")

            act = act_functs[activation]
            self.activation = act[0]
            self.activation_slope = act[1]
        except KeyError:
            print(f"KeyError: {activation} is not a valid activation Key.\nValid keys are {act_functs.keys()}.\n")
            raise(KeyError)

        except Exception as e:
            print(f"{e}:HUH !!!")
            return
        try:
            if backend != "numpy":
                print(f"{backend} has been set on the Network object ")

            self.matmul = matmul_functs[backend]
            self.backend = backend
            
        except KeyError:
            print(f"KeyError: {backend} is not a valid backend key.\nValid keys are {matmul_functs.keys()}.\n")
            raise(KeyError)

        except Exception as e:
            print(f"{e}:HUH !!!")
            return

    # genWeightsBiases generates the weights and biases for the layer based on the provided weight distribution key.
    def genWeightsBiases(self, previousNeuronLength:int, weightDist:str="default") -> None:
        # the shape of the weights is determined by the number of neurons in the previous layer and the number of neurons in the current layer.
        shape = (previousNeuronLength, self.neuronLength)

        # the weights are generated based on the provided weight distribution key. the biases are initialized to zero. change this later. maybe add a bias distribution as well.
        distributions = {
            "default":        lambda : np.random.uniform(-1/shape[1], 1/shape[1], shape),
            "Xavier_Uniform": lambda : np.random.uniform(-np.sqrt(6 / sum(shape)), np.sqrt(6 / sum(shape)), shape),
            "Xavier_Normal":  lambda : np.random.normal(0, np.sqrt(2 / sum(shape)), shape),
            "He_Uniform":     lambda : np.random.uniform(-np.sqrt(6 / shape[0]), np.sqrt(6 / shape[0]), shape),
            "He_Normal":      lambda : np.random.normal(0, np.sqrt(2 / shape[0]), shape),
            "Lecun_Normal":   lambda : np.random.normal(0, np.sqrt(1 / shape[0]), shape),
            "Uniform_Small":  lambda : np.random.uniform(-0.1, 0.1, shape),
            "Zeros":          lambda : np.zeros(shape=shape)
        }

        try:
            self.weights = distributions[weightDist]()

        except KeyError:
            print(f"KeyError: {weightDist} is not a valid weight distribution Key.\nValid keys are {distributions.keys()}.\n")
            raise(KeyError)

        except Exception as e:
            print(f"{e}:HUH !!!")
            return

        self.biases = np.zeros(self.neuronLength)

    # forward takes the input from the previous layer, performs the matrix multiplication with the weights, adds the biases, and applies the activation function to get the output of the layer.
    def forward(self, inp:np.ndarray) -> np.ndarray:
        self.inp = inp
        self.values = self.matmul(inp, self.weights, "forward")
        self.activatedValues = self.activation(self.values)
        return self.activatedValues

    # backward takes the gradient of the loss with respect to the output of the layer and calculates the gradients for the weights, biases, and input of the layer. 
    # It then updates the weights and biases based on the learning rate and returns the gradient of the loss with respect to the input of the layer for use in the backward pass of the previous layer.
    # takes dL_dz which is the gradient of the loss with respect to the output of the layer. it is calculated in the loss function and stored in the network object for use in the backward pass.
    # dl_dz is dependent on how output layer is set up. 
    # if the output layer activation slope is a string, it is assumed to be a loss function and dl_dz is calculated in the loss function. 
    # if its a function, it is assumed to be an activation slope and dl_dz is calculated by multiplying the gradient of the loss with respect to the output of the layer with the activation slope.
    def backward(self, dL_dz:np.ndarray) -> np.ndarray:
        if callable(self.activation_slope): #self.activation_slope is not a string 
            dL_daz = dL_dz * self.activation_slope(self.values)

        else: # is a string
            dL_daz = dL_dz

        self.dl_daz = dL_daz

        # the gradients for the weights, biases, and input of the layer are calculated based on the gradient of the loss with respect to the activated output of the layer and the input to the layer.
        dL_dw = self.matmul(self.inp.T, dL_daz, "backward") / len(self.inp) # the weights are updated based on the average gradient over the batch.
        dL_db = np.mean(dL_daz, axis=0)
        dL_din = self.matmul(dL_daz, self.weights.T, "backward") # the gradient of the loss with respect to the input of the layer is calculated for use in the backward pass of the previous layer.
        self.dl_din = dL_din
        # the weights and biases are updated based on the learning rate and the gradients.
        self.weights -= self.lr * dL_dw
        self.biases -= self.lr * dL_db 
        return dL_din

    # the activation functions and their slopes are defined as separate methods for modularity and ease of use in the forward and backward passes.
    def Relu(self, neuron):
        return np.maximum(neuron,0)
    
    def ReluSlope(self, neuron):
        return (neuron>=0).astype(float)

    def LRelu(self, neuron):     
        return np.where(neuron>=0, neuron, neuron*self.alpha)
    
    def LReluSlope(self, neuron):
        return np.where(neuron>=0, 1, self.alpha)
        
    def sigmoid(self ,neuron):
        return 1 / (1 + np.exp(-neuron))
    
    def SoftMax(self, neuron):
        mx = np.max(neuron, axis=1, keepdims=True)
        e = np.exp(neuron-mx)
        return e/np.sum(e, axis=1, keepdims=True)

    # the matrix multiplication functions are defined as separate methods for modularity and ease of use in the forward and backward passes.
    def numpyMatmul(self, a, b, dir):
        if dir == "forward":
            return a @ b + self.biases
        elif dir == "backward":
            return a @ b

    def openCl(self, a, b, dir):
        if dir == "forward":
            biases = self.biases
        elif dir == "backward":
            biases = np.zeros(self.biases.shape)
            
        output = np.zeros(a.shape[0]*b.shape[1])

        funct_a = a.flatten().astype(np.float32) 
        funct_b = b.flatten().astype(np.float32) 
        funct_biases = biases.astype(np.float32)
        funct_output = output.astype(np.float32)

        self.borrow(funct_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    funct_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    funct_biases.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    funct_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    a.shape[0],
                    b.shape[1],
                    a.shape[1])
        
        return funct_output.reshape(a.shape[0], b.shape[1]).astype(np.float64)

    def cuda(self, a, b):
        self.borrow()

    def show(self):
        attrs = {
            "weights": self.weights,
            "biases": self.biases,
            "inp": self.inp,
            "values": self.values,
            "activated": self.activatedValues,
        }
        for name, val in attrs.items():
            if val is None:
                print(f"{name}: None")
            else:
                import numpy as np
                arr = np.array(val)
                print(f"{name}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} zeros={np.sum(arr==0)}")
        print()

tn = pd.read_csv("./datasets/MNIST_data_train.csv")
tt = pd.read_csv("./datasets/MNIST_data_test.csv")

train_in = tn.drop(["label"], axis=1).to_numpy()
    
train_out = tn["label"].to_numpy()

train = (train_in, train_out)

test_in = tt.drop(["label"], axis=1).to_numpy()
    
test_out = tt["label"].to_numpy()

test = (test_in, test_out)

dataPro = Preprocessor(train, test)
dataPro.features()
dataPro.labels()
train, test = dataPro.data()

a, b = train[0].shape
c, d = train[1].shape

snn = Network([
    Layers(b),
    Layers(b//2, "relu"),
    Layers(b//4, "relu"),
    Layers(d, "SFMX")
])

# snn = Network([
#     Layers(17),
#     Layers(10, "relu"),
#     Layers(38, "relu"),
#     Layers(2, "SFMX")
# ])

# snn = Network([
#     Layers(17),
#     Layers(10, "relu", backend="openCl"),
#     Layers(38, "relu", backend="openCl"),
#     Layers(2, "SFMX")
# ])

snn.Compile(LearningRate=0.1, metrics=["Accuracy", "F1", "Precision", "Recall"], weightDist="He_Uniform")

snn.Fit(train=train, n=10, epochs=10)

snn.Test(test)

snn.plotMetrics()