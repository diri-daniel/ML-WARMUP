import pandas as pd
import numpy
from Experiment.Layers import Layers
from Experiment.Network import Network
from Experiment.Preprocessor import Preprocessor
from Experiment.Network_types import NetworkType

# Note:
# 1. testing is not implemented yet. only training is implemented. - Done
# 2. implement accuracy and other metrics. - Done
# 3. implement opencl and cuda backend for layers. - rewrite .. they keep failing at either extreme values or large datasets.
# 4. add save and reuse functionality.
# 5. maybe add a simple preprocessor extend functionality. 1/2
# 6. maybe code a simple parameter randomizer for testing and experimentation purposes.
# 7. timers for experimentation purposes. maybe make a simple class for this that can be used as a context manager.
# 8. bricked New_pc branch. wont miss it.


tn = pd.read_csv("datasets/MNIST-DIGITS/csv/MNIST_data_train.csv")
tt = pd.read_csv("datasets/MNIST-DIGITS/csv/MNIST_data_test.csv")

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
],"Exp1", NetworkType.Simple_Neural_Network)

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

snn.Fit(train=train, n=10, epochs=35)

snn.Test(test)

snn.save()

print(snn.export()[:2], snn.export()[-1].keys())

snn.plotMetrics()

