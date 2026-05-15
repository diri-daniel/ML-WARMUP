import numpy as np
import pandas as pd
import ctypes
import matplotlib.pyplot as plt

class Network:
    def __init__(self, layers:list)->None:
        self.layers = layers

    def Compile(self, LearningRate:float=0.01, Loss:str="CCE", metrics:list=["Accuracy"], weightDist:str="default", force:bool=False) -> None:

        lastLayerSimplify = self.layers[-1].activation_slope
        if not callable(lastLayerSimplify) and not force:
            Loss = lastLayerSimplify
            print(f"Auto-set loss to {Loss}. Set force=True to override.")

        elif force:
            print(f"Warning: Loss function {Loss} is forced. Its not an error but please confirm {Loss} is appropriate for last layer slope")

        else:
            print(f"Warning: last layer is a function. Its not an error but please confirm {Loss} is appropriate")

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

        for i in range(1, len(self.layers)):
            self.layers[i].genWeightsBiases(self.layers[i-1].neuronLength, weightDist=weightDist)
            self.layers[i].lr = LearningRate

    def Forward(self, train:np.ndarray) -> None:
        inp = train
        for layer in self.layers[1:]:
            inp = layer.forward(inp)

        self.output = inp
        

    def Backward(self):
        dL_dn = self.dL_do
        for layer in reversed(self.layers[1:]):
            dL_dn = layer.backward(dL_dn)

    def Fit(self, train, epochs):
        for epoch in range(epochs):
            self.Forward(train[0])
            self.dL_do = self.loss(train[1])# get this after running the loss function should not be hard
            self.Backward()

    def catCrossEnt(self, target):
        x = self.output - target
        self.Loss = -np.mean(np.sum(target * np.log(np.clip(self.output, 1e-7, 1)), axis=1))
        return x

    def binCrossEnt(self, target):
        x = self.output - target
        self.Loss = -np.mean(target * np.log(np.clip(self.output, 1e-7, 1)) + (1 - target) * np.log(np.clip(1 - self.output, 1e-7, 1)))
        return x

    def accuracy(self):
        pass

class Layers:
    def __init__(self, size:int, activation:str="relu", alpha:float=0.1, backend:str="numpy"):
        
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
        
        self.neuronLength = size
        self.alpha = alpha
        try:
            act = act_functs[activation]
            self.activation = act[0]
            self.activation_slope = act[1]
        except KeyError:
            print(f"KeyError: {activation} is not a valid activation Key.\nValid keys are {act_functs.keys()}.\n")
            return

        except Exception as e:
            print(f"{e}:HUH !!!")
            return
        try:
            self.matmul = matmul_functs[backend]
            
        except KeyError:
            print(f"KeyError: {backend} is not a valid backend Key.\nValid keys are {matmul_functs.keys()}.\n")
            return

        except Exception as e:
            print(f"{e}:HUH !!!")
            return

    def genWeightsBiases(self, previousNeuronLength:int, weightDist:str="default") -> None:
        shape = (previousNeuronLength, self.neuronLength)
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
            return

        except Exception as e:
            print(f"{e}:HUH !!!")
            return

        self.biases = np.zeros(self.neuronLength)

    def forward(self, inp:np.ndarray) -> np.ndarray:
        self.inp = inp
        self.values = self.matmul(inp, self.weights)
        self.activatedValues = self.activation(self.values)
        return self.activatedValues

    def backward(self, dL_dz:np.ndarray) -> np.ndarray:
        if not callable(self.activation_slope): #self.activation_slope is not a string
            dL_daz = dL_dz * self.activation_slope(self.values)

        else: # is a string
            dL_daz = dL_dz

        dL_dw = self.inp.T @ dL_daz / len(self.inp)
        dL_db = np.mean(dL_daz, axis=0)
        dL_din = self.matmul(dL_daz, self.weights.T)

        self.weights -= self.lr * dL_dw
        self.biases -= self.lr * dL_db 
        return dL_din

    def Relu(self, neuron):
        return np.maximum(neuron,0)
    
    def ReluSlope(self, neuron):
        return (neuron>=0).astype(float)

    def LRelu(self, neuron):
        print(f"Warning: alpha already set. Set layers(alpha) at initialization")
        return np.where(neuron>=0, neuron, neuron*self.alpha)
    
    def LReluSlope(self, neuron):
        print(f"Warning: alpha already set. Set layers(alpha) at initialization")
        return np.where(neuron>=0, 1, self.alpha)
        
    def sigmoid(self ,neuron):
        return 1 / (1 + np.exp(-neuron))
    
    def SoftMax(self, neuron):
        mx = np.max(neuron, axis=1, keepdims=True)
        e = np.exp(neuron-mx)
        return e/np.sum(e, axis=1, keepdims=True)

    def numpyMatmul(self, a, b):
        return a @ b

    def openCl(self, a, b):
        pass


snn = Network([
    Layers(6, None, None),
    Layers(10),
    Layers(38)
])

snn.Compile(90, Loss="CCE", metrics=[])
