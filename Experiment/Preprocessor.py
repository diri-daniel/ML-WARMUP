import numpy as np

# Note:
# 1. testing is not implemented yet. only training is implemented. - Done
# 2. implement accuracy and other metrics. - Done
# 3. implement opencl and cuda backend for layers. - rewrite .. they keep failing at either extreme values or large datasets.
# 4. add save and reuse functionality.
# 5. maybe add a simple preprocessor extend functionality. 1/2
# 6. maybe code a simple parameter randomizer for testing and experimentation purposes.
# 7. timers for experimentation purposes. maybe make a simple class for this that can be used as a context manager.
# 8. bricked New_pc branch. wont miss it.


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