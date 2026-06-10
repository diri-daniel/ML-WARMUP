import numpy as np
import ctypes

# Note:
# 1. testing is not implemented yet. only training is implemented. - Done
# 2. implement accuracy and other metrics. - Done
# 3. implement opencl and cuda backend for layers. - rewrite .. they keep failing at either extreme values or large datasets.
# 4. add save and reuse functionality.
# 5. maybe add a simple preprocessor extend functionality. 1/2
# 6. maybe code a simple parameter randomizer for testing and experimentation purposes.
# 7. timers for experimentation purposes. maybe make a simple class for this that can be used as a context manager.
# 8. bricked New_pc branch. wont miss it.


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