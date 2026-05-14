import numpy as np
import pandas as pd
import ctypes
import matplotlib.pyplot as plt

class Network:
    def __init__(self, layers:list)->None:
        self.layers = layers

    def Compile(self, LearningRate:float=0.01, Loss:str="CCE", metrics:list=["Accuracy"], weightDist:str="default") -> None:
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
            self.dL_do = self.loss()# get this after running the loss function should not be hard
            self.Backward()

    def catCrossEnt(self):
        pass

    def binCrossEnt(self):
        pass

    def accuracy(self):
        pass

class Layers:
    def __init__(self, size:int, activation:str="relu", backend:str="numpy"):
        # add try catch and error handling, SFMX doesnt need slope but still
        act_functs = {
            "relu" : [self.Relu, self.ReluSlope],
            "lrelu" : [self.LRelu, self.LReluSlope],
            "sigmoid" : [self.sigmoid, self.sigmoidSlope],
            "SFMX" : [self.SoftMax, self.SoftMaxSlope]
        }
        matmul_functs = {
            "numpy" : self.numpyMatmul,
            "openCl" : self.openCl 
        }
        self.neuronLength = size
        self.activation = act_functs.get(activation, None)[0]
        self.activation_slope = act_functs.get(activation, None)[1]
        self.matmul = matmul_functs.get(backend, None)

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

    def backward(self, dL_dn:np.ndarray) -> np.ndarray:
        self.dL_dn = dL_dn
        dL_dz = dL_dn * self.activation_slope(self.values)

        dL_dw = self.inp.T @ dL_dz / len(self.inp)
        dL_db = np.mean(dL_dz, axis=0)
        dL_din = self.matmul(dL_dz, self.weights.T)

        self.weights -= self.lr * dL_dw
        self.bias -= self.lr * dL_db 
        return dL_din

    def Relu(self):
        pass

    def LRelu(self):
        pass

    def sigmoid(self):
        pass

    def SoftMax(self):
        pass

    def numpyMatmul(self, a, b):
        pass

    def openCl(self, a, b):
        pass


snn = Network([
    Layers(6, None, None),
    Layers(10),
    Layers(38)
])

snn.Compile(90, Loss="CCE", metrics=[])
