from .Network_types import NetworkType
import numpy as np
import datetime
import os

# Note:
# 1. testing is not implemented yet. only training is implemented. - Done
# 2. implement accuracy and other metrics. - Done
# 3. implement opencl and cuda backend for layers. - rewrite .. they keep failing at either extreme values or large datasets.
# 4. add save and reuse functionality. - done
# 5. maybe add a simple preprocessor extend functionality. 1/2
# 6. maybe code a simple parameter randomizer for testing and experimentation purposes.
# 7. timers for experimentation purposes. maybe make a simple class for this that can be used as a context manager.
# 8. bricked New_pc branch. wont miss it.
# 9. add import and export functionality for db uses. - done


class Network:
    def __init__(self, layers:list, name:str, type:NetworkType=NetworkType.Simple_Neural_Network, helper="./Helper/")->None:
        # layers should be a list of Layers objects.
        # The first layer should be the input layer and the last layer should be the output layer. 
        # The input layer is not used for calculations but is used to set the input shape for the first hidden layer. 
        # The output layer is used to set the output shape for the last hidden layer. 
        # The hidden & output layers are used for calculations and can have any activation function and weight distribution.
        self.layers = layers
        self.helper = helper

        try:
            self.name = name
            self.type = type

        except Exception as e:
            raise(e)

    def initOpenCl(self):
        import ctypes
        self.lib = ctypes.CDLL(self.helper+"main.dll")
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
        precisions = []
        for c in range(target.shape[1]):
            tp = np.sum((predicted == c) & (actual == c))
            fp = np.sum((predicted == c) & (actual != c))
            precisions.append(tp / (tp + fp + 1e-7))
        return np.mean(precisions)

    def recall(self, target):
        predicted = np.argmax(self.output, axis=1)
        actual = np.argmax(target, axis=1)
        recalls = []
        for c in range(target.shape[1]):
            tp = np.sum((predicted == c) & (actual == c))
            fn = np.sum((predicted != c) & (actual == c))
            recalls.append(tp / (tp + fn + 1e-7))
        return np.mean(recalls)

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
        import matplotlib.pyplot as plt
        x = np.arange(start=0, step=1, stop= len(self.pLoss))
        y = self.pLoss
        plt.plot(x, y, label="Loss")
        for metric in self.metrics:
            plt.plot(x, self.sMV[metric], label=metric)
        plt.legend()
        plt.show()

    def getDetails(self):
        pass

    def _writeReadme(self, path, content):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.writelines(content)

    def _writeNames(self, base):
        nameMD = base + "/" + self.type + "/Names.md"
        os.makedirs(os.path.dirname(nameMD), exist_ok=True)

        if not os.path.exists(nameMD):
            with open(nameMD, "w") as n:
                n.writelines([f"{self.type} Names.\n", "\n"])

        with open(nameMD, "r+") as n:
            namesmd = n.readlines()

        if len(namesmd) <= 2 or f"- {self.name}\n" not in namesmd[2:]:
            with open(nameMD, "a") as n:
                n.write(f"- {self.name}\n")

    def save(self):
        day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        now = datetime.datetime.now()
        cache = "Cache/" + self.type + "/" + self.name + "/" + str(now.year) +"/"+ month[now.month - 1] +"/"+ day[now.weekday()] + " " + str(now.day) +"/"+ str(now.hour) +"-"+ str(now.minute) +"-"+ str(now.second)
        curr = "Outputs/" + self.type + "/" + self.name

        i = 1
        for layers in self.layers[1:]:
            if i < len(self.layers) - 1:
                tag = f"h{i}"
            else:
                tag = "out"

            temp1_w = cache + "/" + tag + "/" + "weights.npy"
            temp1_b = cache + "/" + tag + "/" + "biases.npy"
            temp2_w = curr + "/" + tag + "/" + "weights.npy"
            temp2_b = curr + "/" + tag + "/" + "biases.npy"

            os.makedirs(os.path.dirname(temp1_w), exist_ok=True)
            os.makedirs(os.path.dirname(temp1_b), exist_ok=True)
            os.makedirs(os.path.dirname(temp2_w), exist_ok=True)
            os.makedirs(os.path.dirname(temp2_b), exist_ok=True)

            with open(temp1_w, "wb") as a:
                np.save(a, layers.weights)
            with open(temp2_w, "wb") as b:
                np.save(b, layers.weights)
            with open(temp1_b, "wb") as c:
                np.save(c, layers.biases)
            with open(temp2_b, "wb") as d:
                np.save(d, layers.biases)

            i+=1

        self._writeNames("Outputs")
        self._writeNames("Cache")

        # details = self.getDetails()
        # self._writeReadme(curr + "/README.md", details)
        # self._writeReadme(cache + "/README.md", details)

    def load(self, name:str=None, type:NetworkType=None):
        if not name: name = self.name
        if not type: type = self.type
        
        curr = "Outputs/" + type + "/" + name
        if not os.path.exists(curr):
            raise FileNotFoundError(f"Model not found: {curr}")
        
        i = 1
        for layers in self.layers[1:]:
            tag = f"h{i}" if i < len(self.layers) - 1 else "out"
            
            w_path = curr + "/" + tag + "/weights.npy"
            b_path = curr + "/" + tag + "/biases.npy"
            
            with open(w_path, "rb") as f:
                layers.weights = np.load(f)
            with open(b_path, "rb") as f:
                layers.biases = np.load(f)
            i += 1

    def export(self):
        model_data = {}
        i = 1
        for layers in self.layers[1:]:
            tag = f"h{i}" if i < len(self.layers) - 1 else "out"
            model_data[tag] = {
                "weights": layers.weights.tobytes(),
                "biases": layers.biases.tobytes(),
                "shape_w": layers.weights.shape,
                "shape_b": layers.biases.shape
            }
            i += 1
        return (self.type, self.name, model_data)
    
    def import_model(self, model_data):
        i = 1
        for layers in self.layers[1:]:
            tag = f"h{i}" if i < len(self.layers) - 1 else "out"
            data = model_data[tag]
            
            layers.weights = np.frombuffer(data["weights"]).reshape(data["shape_w"]).copy()
            layers.biases = np.frombuffer(data["biases"]).reshape(data["shape_b"]).copy()
            i += 1

