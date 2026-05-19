import numpy as np
import pandas as pd
import ctypes
import matplotlib.pyplot as plt

# Note:
# 1. testing is not implemented yet. only training is implemented.
# 2. implement accuracy and other metrics later. for now, just implement loss and make sure it decreases over epochs.
# 3. implement opencl backend for layers.
# 4. maybe code a simple parameter randomizer for testing and experimentation purposes.
# 5. timers for experimentation purposes. maybe make a simple class for this that can be used as a context manager.
# 6. maybe add a simple preprocessor attached to network.compile().


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

        # set the loss function and metric functions based on the provided keys.
        loss_functs = {
            "CCE" : self.catCrossEnt,
            "BCE" : self.binCrossEnt
        }
        metric_functs = {
            "Accuracy" : self.accuracy
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
            self.metrics = []
            for x in metrics:
                self.metrics.append(metric_functs[x])

        except KeyError:
            print(f"KeyError: {x} is not a valid metric Key.\nValid keys are {metric_functs.keys()}.\n")
            return

        except Exception as e:
            print(f"{e}:HUH !!!")
            return

        # generate weights and biases for all layers except the input layer. set learning rate for all layers.
        for i in range(1, len(self.layers)):
            self.layers[i].genWeightsBiases(self.layers[i-1].neuronLength, weightDist=weightDist)
            self.layers[i].lr = LearningRate

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
        dL_dn = self.dL_do # the gradient of the loss with respect to the output of the network. it is calculated in the loss function and stored in the network object for use in the backward pass.

        # simple backward pass. the gradient of the loss with respect to the output of the network is passed through the network layer by layer in reverse order.
        for layer in reversed(self.layers[1:]):
            dL_dn = layer.backward(dL_dn)

    # Fit should be called after Compile. 
    # It takes the training data and the number of epochs and trains the network by calling Forward and Backward for each epoch. 
    # It also stores the loss for each epoch in the network object for use in plotting the loss curve.
    def Fit(self, train, epochs=100):
        #targets = self.encode(train[1]) fix this later. only encode if requested and appropriate parameters are set. for now, just assume the targets are already encoded.
        self.pLoss = [] # stores loss for each epoch.
        targets = train[1] # change this later. only encode if requested and appropriate parameters are set. for now, just assume the targets are already encoded.
        self.acc = [] # stores accuracy for each epoch. fix this later. only calculate if requested and appropriate parameters are set. for now, just assume accuracy is always calculated.
        # simple training loop.
        for epoch in range(epochs):
            self.Forward(train[0])
            self.dL_do = self.loss(targets) # precariously stores the gradient of the loss with respect to the output of the network for use in the backward pass. change this later. might be made more modular.
            self.Backward()
            self.pLoss.append(float(self.Loss))
            self.acc.append(float(self.accuracy(targets)))

    # the loss functions return the gradient of the loss with respect to the output of the network and also store the loss in the network object for use in plotting the loss curve.
    def catCrossEnt(self, target):
        x = self.output - target
        self.Loss = -np.mean(np.sum(target * np.log(np.clip(self.output, 1e-7, 1)), axis=1))
        return x

    def binCrossEnt(self, target):
        x = self.output - target
        self.Loss = -np.mean(target * np.log(np.clip(self.output, 1e-7, 1)) + (1 - target) * np.log(np.clip(1 - self.output, 1e-7, 1)))
        return x

    # fix later but onehot encodes actual outputs. should be made more automatic.
    def encode(self, targets):
        arr = np.zeros((len(targets), self.layers[-1].neuronLength))
        arr[np.arange(len(targets)), targets.astype(int)] = 1
        return arr

    # implemented but unused.
    def accuracy(self, target):
        predicted = np.argmax(self.output, axis=1)
        actual = np.argmax(target, axis=1)
        return np.mean(predicted == actual)
    
class Preprocessor():
    def __init__(self, data:tuple, output:str="onehot", split:float=0.8)->None:
        pass

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
            "numpy" : self.numpyMatmul,
            "openCl" : self.openCl 
        }
        
        # size is the number of neurons in the layer. activation is the activation function for the layer. 
        # alpha is the slope for leaky relu. 
        # backend is the backend for matrix multiplication.
        self.neuronLength = size
        self.alpha = alpha
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
            self.matmul = matmul_functs[backend]
            
        except KeyError:
            print(f"KeyError: {backend} is not a valid backend Key.\nValid keys are {matmul_functs.keys()}.\n")
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
        self.values = self.matmul(inp, self.weights) + self.biases
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

        # the gradients for the weights, biases, and input of the layer are calculated based on the gradient of the loss with respect to the activated output of the layer and the input to the layer.
        dL_dw = self.inp.T @ dL_daz / len(self.inp) #change this later. use self.matmul(). might have to look at the dot products of I.T * dL/daz and dL/daz.T and I
        dL_db = np.mean(dL_daz, axis=0)
        dL_din = self.matmul(dL_daz, self.weights.T)

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
    def numpyMatmul(self, a, b):
        #print(a.shape, b.shape)
        return a @ b

    def openCl(self, a, b):
        pass

    def cuda(self, a, b):
        pass


snn = Network([
    Layers(17),
    Layers(10, "relu"),
    Layers(38, "relu"),
    Layers(2, "SFMX")
])

snn.Compile(LearningRate=0.1)

tn = pd.read_csv("./datasets/data-1-onehot-train.csv")
tt = pd.read_csv("./datasets/data-1-onehot-test.csv")

    # print(f"Train data spread:\n{tn["class"].value_counts()}")
    # print(f"Test data spread:\n{tt["class"].value_counts()}")

train_in = tn.drop(["class"], axis=1).to_numpy()
    # train_in = np.array([[int(w.strip("b'")) for w in v] for v in train_in], dtype=np.int32)
train_out = np.where(tn["class"].to_numpy() == "b'0'", 0, 1)
train_out = np.array([[1, 0] if x == 0 else [0, 1] for x in train_out])

train = (train_in, train_out)

test_in = tt.drop(["class"], axis=1).to_numpy()
    # test_in = np.array([[int(w.strip("b'")) for w in v] for v in test_in], dtype=np.int32)
test_out = tt["class"].to_numpy()

test = (test_in, test_out)

snn.Fit(train=train, epochs=1000)

x = np.arange(start=0, step=1, stop= len(snn.pLoss))
y = snn.pLoss
y2 = snn.acc

print(f"Starting Loss: {y[0]}, Ending Loss: {y[-1]}")
print(f"Starting Accuracy: {y2[0]}, Ending Accuracy: {y2[-1]}")

plt.plot(x, y)
plt.plot(x, y2)
plt.show()