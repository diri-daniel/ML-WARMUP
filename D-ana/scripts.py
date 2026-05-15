from scipy.io import arff
import ctypes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import time

class Node():
    def __init__(self, data, root=True, level=0, side=None, depth=None):
        self.level = level
        self.is_root = root
        self.data = data
        self.le_feature, self.le_val, self.is_leaf, self.le_class, self.le_feature_ind = self.find_vals(data)
        
        if not depth or not(depth == level):
            if  (not self.is_leaf):
                # print("level | feature | feature value | is leaf | class | side")
                # print(self.level, self.le_feature, self.le_val, self.is_leaf, self.le_class, side, sep=" | ")
                self.left_data = data[data[self.le_feature] == self.le_val]
                self.right_data = data[data[self.le_feature] != self.le_val]
                self.left = Node(self.left_data, root=False, level= self.level + 1, side="left")
                self.right = Node(self.right_data, root=False, level = self.level + 1, side="right")

            # else:
                # print(f"leaf at: {level}, {side} with {self.le_feature} == {self.le_val} and class {self.le_class}")

    def find_vals(self, df):
        features = df.drop(['class'], axis=1).columns
        colums_uval = {v : df[v].unique() for v in df.columns}

        # pp.pprint(colums_uval)

        vals = []
        weights = []

        for f, v in colums_uval.items():
            if f != 'class':
                cl_spread = []
                spread = []
                for a in v:
                    cl_spread.append([(len(df[(df[f] == a) & (df['class'] == x)])/len(df) * 100) for x in df['class'].unique()])
                    spread.append((len(df[df[f] == a])/len(df) * 100))

                #pp.pprint(f"{f} : {cl_spread}")
                vals.append(cl_spread)
                weights.append(spread)

        e_raw = [[self.entropy(d) for d in v] for v in vals]

        e_vals = []
        for i, v in enumerate(e_raw):
            e_vals.append(sum(np.array(weights[i]) * np.array(v)))

        le_feature_ind = np.array(e_vals).argmin()
        le_feature = features[le_feature_ind]
        le_val_ind = np.array(e_raw[le_feature_ind]).argmin()
        le_val = colums_uval[le_feature][le_val_ind]

        leaf = False
        le_class = None

        #print(f"min entropy: {int(min(e_vals))}")

        min_samples = 5

        if len(df['class'].unique()) == 1:
            leaf = True
            le_class = df['class'].unique()[0]
        elif len(df) < min_samples:
            leaf = True
            le_class = df['class'].value_counts().index[0]

        # pp.pprint(vals)
        # pp.pprint(e_raw)
        # pp.pprint(e_vals)

        # print(le_feature)
        # print(le_val)

        return (le_feature, le_val, leaf, le_class, le_feature_ind)

    def entropy(self, dist):
            total = sum(dist)
            if total == 0:
                return 0
            
            probs = [p/total for p in dist if p > 0]

            return -sum([p * np.log2(p) for p in probs])
    
    def query(self, inp):
        ind = self.le_feature_ind
        while True:
            if self.le_class:
                return self.le_class
            elif inp[ind] == self.le_val:
                return self.left.query(inp)
            elif inp[ind] != self.le_val:
                return self.right.query(inp)
            
class Network: 

    def __init__(self, train, test, network):
        print(network)
        self.train_in, self.train_out = train
        self.classes = np.unique(self.train_out)
        self.train_out_ind = np.array([int(np.argwhere(self.classes==v)[0][0]) for v in self.train_out])
        #print(self.classes)
        self.batchLength = len(self.train_in)
        self.test_in, self.test_out = test
        self.TestLength = len(self.test_in)
        self.test_out_ind = np.array([int(np.argwhere(self.classes==v)[0][0]) for v in self.test_out])
        self.weights = [
            np.random.randn(network[i-1], network[i]) * np.sqrt(2 / network[i-1])
            for i in range(1, len(network))
        ]
        self.bias = [np.zeros(shape=(1,network[i]), dtype=np.float64) for i in range(1,len(network))]

        self.actFuncs = [self.RELU, self.ABV, self.SFTMX]

        # pp.pprint(self.weights)
        # pp.pprint(self.bias)

        self.lib = ctypes.CDLL("./D-ops/main.dll")
        self.lib.init_opencl()

        self.lib.snn_forward.argtypes  = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]

        self.lib.init_snn()

    def forward(self):
        self.neuron_outs = []
        for t, v in enumerate(self.weights):
            if t == 0:
                inp = np.astype(self.train_in.flatten(), np.float32)
                
            else:
                inp = act_out

            weights = np.astype(self.weights[t].flatten(), np.float32)
            bias = np.astype(self.bias[t], np.float32)
            out = np.zeros(self.batchLength*len(self.weights[t][0]), dtype=np.float32)
            self.lib.snn_forward(
                inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.batchLength,
                len(self.weights[t][0]),
                len(self.weights[t])
            )

            if t == len(self.weights) - 1:
                act_out = np.array([self.actFuncs[2](
                    out[i*len(self.weights[t][0]): i*len(self.weights[t][0]) + len(self.weights[t][0])]
                ) for i in range(self.batchLength)])
                # print(out)
            else:
                # print("mid")
                act_out = self.actFuncs[0](out)


                # print(out.reshape((self.batchLength, len(self.weights[t][0]))))

            self.neuron_outs.append([out, act_out])

        self.output = act_out
    
    def RELU(self, out):
        return np.maximum(0, np.array(out))
    
    def ABV(self, out):
        return np.absolute(np.array(out))
    
    def minMax(self, out):
        mn = min(out)
        mx = max(out)
        return [(x-mn)/(mx-mn) for x in out]
    
    def SFTMX(self, out):
        mx = max(out)
        s = sum([pow(np.e, x-mx) for x in out])
        return [pow(np.e, x-mx)/s for x in out]
    
    def metrics(self, actual):
        weight = self.output.max(axis=1)
        inds = self.output.argmax(axis=1)

        out = actual
        #print(out)
        totalCorrect = [[1, float(1 * weight[i])] if out[i] == v else [0, 0.0] for i, v in enumerate(inds) ]
        self.accuracy, self.weightedAccuracy = tuple(np.average(np.array(totalCorrect),axis=0))
        self.Loss = self.CCE(self.encode(actual), self.output)
        #print(self.accuracy, self.weightedAccuracy, self.Loss, sep=", ")
        return np.array([self.accuracy, self.weightedAccuracy, self.Loss])

    def BCE(self, act, pred):
        eps = 1e-7
        pred = np.clip(pred, eps, 1 - eps)
        return -np.mean(act*np.log(pred) + (1-act)*np.log(1-pred))
    
    def CCE(self, act, pred):
        eps = 1e-7
        pred = np.clip(pred, eps, 1 - eps)
        return -np.mean(np.sum(act * np.log(pred), axis=1))
    
    def encode(self, actual):
        arr = np.zeros_like(self.output)
        for i, v in enumerate(actual):
            arr[i][v] = 1

        return arr
    
    def backward(self, lr=0.1):
        n = len(self.weights)
        dLDO = []
        for i in range(n - 1,-1, -1):
            if i == n -1:
                deltaLossDeltaOut = np.astype(self.CE_SFTMX_slope(self.output, self.encode(self.train_out_ind)).flatten(), np.float32)

            else:
                deltaLossDeltaOut = np.astype(out * self.RELU__slope(self.neuron_outs[i][0]), np.float32)
                # print(f"i={i}, neuron_outs[i][0] max={self.neuron_outs[i][0].max()}, min={self.neuron_outs[i][0].min()}")
                # print(f"out max={out.max()}, out min={out.min()}")
                # print(f"after multiply max={(out * self.RELU__slope(self.neuron_outs[i][0])).max()}")
                # print(f"trainbuf size={self.batchLength * len(np.transpose(self.weights[i]))}")
                # print(f"deltaLossDeltaOut actual size={len(deltaLossDeltaOut)}")
                # print(np.astype(np.transpose(self.weights[i]).flatten(), np.float32)[:20])

            dLDO.append(np.array(deltaLossDeltaOut).astype(np.float64))

            # print("deltaLossDeltaOut", deltaLossDeltaOut.shape, deltaLossDeltaOut.max(), deltaLossDeltaOut.min())
            # print("weights shape", np.transpose(self.weights[i]).shape)
            # print("out shape expected", self.batchLength, len(np.transpose(self.weights[i])[0]))
            # print("deltaLossDeltaOut values", deltaLossDeltaOut[:10])
            # print("weights", np.transpose(self.weights[i])[:5])
            # print("neuron length", len(np.transpose(self.weights[i])[0]))  # neuronLength
            # print("input length",len(np.transpose(self.weights[i])))      # inputLength
            
            weights = np.astype(np.transpose(self.weights[i]).flatten(), np.float32)
            bias = np.astype(np.zeros_like(self.bias[i]), np.float32)
            out = np.zeros(self.batchLength*len(np.transpose(self.weights[i])[0]), dtype=np.float32)
            self.lib.snn_forward(
                deltaLossDeltaOut.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.batchLength,
                len(np.transpose(self.weights[i])[0]),
                len(np.transpose(self.weights[i]))
            )
            # out = (deltaLossDeltaOut.reshape(self.batchLength, len(self.weights[i][0])) @ self.weights[i].T).flatten()
            

        for i in range(n):
            if i == 0:
                inp = np.transpose(np.astype(self.train_in, np.float64))

            else:
                inp = np.transpose(np.reshape(self.neuron_outs[i-1][1], (self.batchLength, len(self.weights[i-1][0]))))

            grad = (inp.astype(np.float64) @ np.reshape(dLDO[n-(i+1)], (self.batchLength, len(self.weights[i][0]))).astype(np.float64)) / self.batchLength
            self.weights[i] -= (lr * grad).astype(np.float32)
            bias_grad = np.sum(
                np.reshape(dLDO[n-(i+1)], (self.batchLength, len(self.weights[i][0]))).astype(np.float64), 
                axis=0
            ) / self.batchLength
            self.bias[i] -= (lr * bias_grad).astype(np.float32)

            
        # print(dLDO)

    def CE_SFTMX_slope(self, pred, act):
        return pred - act
    
    def RELU__slope(self, out):
        return (np.array(out) >= 0).astype(float)
    
    def Learn(self, epoch=3000, lr=0.1):
        for i in range(epoch):
            self.forward()
            x = self.metrics(self.train_out_ind)
            self.backward(lr=lr)
            if(i%(epoch/3) == 0):
                print(f"epoch {i}: {x}")    
            # print([w.max() for w in self.weights])
            # print([w.min() for w in self.weights])
            # print([b.max() for b in self.bias])
            # print([b.min() for b in self.bias])
        

    def save(self):
        for i, v in enumerate(self.weights):
            pd.DataFrame(v).to_csv(f"./Outputs/SNN/Weights/h{i+1}.csv",index=None)
            pd.DataFrame(self.bias[i]).to_csv(f"./Outputs/SNN/Biases/h{i+1}.csv",index=None)
        self.clean()
        
    def Test(self):
        for t, v in enumerate(self.weights):
            if t == 0:
                inp = np.astype(self.test_in.flatten(), np.float32)
                
            else:
                inp = act_out

            weights = np.astype(self.weights[t].flatten(), np.float32)
            bias = np.astype(self.bias[t], np.float32)
            out = np.zeros(self.TestLength*len(self.weights[t][0]), dtype=np.float32)
            self.lib.snn_forward(
                inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.TestLength,
                len(self.weights[t][0]),
                len(self.weights[t])
            )

            if t == len(self.weights) - 1:
                act_out = np.array([self.actFuncs[2](
                    out[i*len(self.weights[t][0]): i*len(self.weights[t][0]) + len(self.weights[t][0])]
                ) for i in range(self.TestLength)])
                # print(out)
            else:
                # print("mid")
                act_out = self.actFuncs[0](out)
        self.output = act_out
        print(self.metrics(self.test_out_ind))
        if self.accuracy > 0.99:
            self.save()

    def clean(self):
        self.lib.cleanup()
            



def s1():
    data, = arff.loadarff("C:/Users/dirid/Desktop/dev/projects/machine learning/datasets/phpAyyBys.arff")
    df = pd.DataFrame(data)
    df.to_csv("./datasets/data-1.csv", index=None)
    print("done")

def s2():
    df = pd.read_csv("datasets/data-1.csv")
    [print(df[x].value_counts(ascending=True)) for x in df.columns]
    [print(df[x].dtype) for x in df.columns]

def s3():
    df = pd.read_csv("datasets/data-1.csv")
    df = df.apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    dfSort = df.sort_values(by=[x for x in df.columns])
    dfSort.to_csv("./datasets/data-1-sorted.csv", index=None)

def s4():
    df = pd.read_csv("./datasets/data-1-sorted.csv")
    classes = df['class'].unique()
    #print(classes)

    grouped = df.groupby('class')
    groups = grouped.groups

    dfDict = {cl : df.drop(['class'],axis=1).loc[groups[cl]] for cl in classes}

    data = {cl : {a : d[a].value_counts().to_dict() for a in d.columns} for cl , d in dfDict.items()}
    pp.pprint(data)
    #print(df.drop(['class'],axis=1).loc[groups["b'0'"]])
    attrs = list(next(iter(data.values())).keys())

    fig, axes = plt.subplots(1, len(attrs), figsize=(20, 5))

    for i, attr in enumerate(attrs):
        ax = axes[i]
        all_vals = sorted(set(v for cl in classes for v in data[cl][attr].keys()))
        x = np.arange(len(all_vals))
        width = 0.35

        for j, cl in enumerate(classes):
            counts = [data[cl][attr].get(v, 0) for v in all_vals]
            ax.bar(x + j * width, counts, width, label=cl)

        ax.set_title(attr)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(all_vals)
        ax.legend()

    plt.tight_layout()
    plt.show()

def s5():
    df = pd.read_csv("datasets/data-1-onehot.csv")
    l = len(df)
    inds = np.arange(start=0, stop=l)
    np.random.shuffle(inds)
    
    # print(inds)

    r = 0.7

    df.loc[list(inds[:int(l*r)])].to_csv("./datasets/data-1-onehot-train.csv", index=None)
    df.loc[list(inds[int(l*r):])].to_csv("./datasets/data-1-onehot-test.csv", index=None)

def s6(k):
    ocl_overhead_start = time.time()
    lib = ctypes.CDLL("./D-ops/main.dll")

    lib.init_opencl()
    
    lib.init_knn.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int
    ]

    lib.knn_distances.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    data_overhead_start = time.time()
    tn = pd.read_csv("./datasets/data-1-train.csv")
    tt = pd.read_csv("./datasets/data-1-test.csv")

    print(f"Train data spread:\n{tn["class"].value_counts()}")
    print(f"Test data spread:\n{tt["class"].value_counts()}")

    n_attr = len(tn.drop(['class'], axis=1).columns)

    training = tn.drop(["class"], axis=1).to_numpy().flatten()
    training = np.array([int(v.strip("b'")) for v in training], dtype=np.int32)
    labels = tn['class'].to_numpy()
    n_samples = len(training)//n_attr

    lib.init_knn(
        training.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n_attr,
        n_samples
    )
    
    query_in = tt.drop(['class'],axis=1).to_numpy()
    query_out = tt['class'].to_list()
    query_pred = []
    query_in = np.array([[int(v.strip("b'")) for v in a] for a in query_in])

    # pp.pprint(training)
    # pp.pprint(query_in)
    # pp.pprint(query_out)
    train_test_start = time.time()

    for i, v in enumerate(query_in):

        distances = np.zeros(n_samples, dtype=np.float32)

        lib.knn_distances(
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_attr,
            n_samples
        )

        indices = np.argsort(distances)[:k]
        neighbors = labels[indices]
        votes = np.unique_counts(neighbors)
        query_pred.append(votes.values[votes.counts.argmax()])

    end = time.time()

    cor = 0
    print(f"k = {k} \n")
    print("Failed predictions \n")
    for i, v in enumerate(query_pred):
        if query_pred[i] == query_out[i]:
            cor += 1

        else:
            print(query_pred[i], query_out[i], i, sep=" ")

    print(f"Total:{end-ocl_overhead_start}", f"Train-Test loop: {end-train_test_start}", 
          f"Data overhead: {train_test_start-data_overhead_start}", 
          f"OpenCl Overhead: {data_overhead_start-ocl_overhead_start}", sep="\n")
    print(f"Accuracy:{cor/len(query_pred) * 100}")
    print(f"Predicted value spread:\n{pd.Series(query_pred).value_counts()}")
    print()
    print()
    
def s7():
    tn = pd.read_csv("./datasets/data-1-train.csv")
    tt = pd.read_csv("./datasets/data-1-test.csv")

    #print(f"Train data spread:\n{tn["class"].value_counts()}")
    #print(f"Test data spread:\n{tt["class"].value_counts()}")

    training = tn

    DT = Node(training)

    query_in = tt.drop(['class'],axis=1).to_numpy()
    query_out = tt['class'].to_list()
    query_pred = []
    
    for v in query_in:
        query_pred.append(DT.query(v))

    cor = 0
    #print("Failed predictions \n")
    for i, v in enumerate(query_pred):
        f = "failed"
        if query_pred[i] == query_out[i]:
            cor += 1
            f = "not failed"

        #print(query_pred[i], query_out[i], i, f, sep=" ")

    print(f"Accuracy:{cor/len(query_pred) * 100}")
    print(f"Predicted value spread:\n{pd.Series(query_pred).value_counts()}")

def s8():

    tn = pd.read_csv("./datasets/data-1-onehot-train.csv")
    tt = pd.read_csv("./datasets/data-1-onehot-test.csv")

    # print(f"Train data spread:\n{tn["class"].value_counts()}")
    # print(f"Test data spread:\n{tt["class"].value_counts()}")

    train_in = tn.drop(["class"], axis=1).to_numpy()
    # train_in = np.array([[int(w.strip("b'")) for w in v] for v in train_in], dtype=np.int32)
    train_out = tn["class"].to_numpy()

    train = (train_in, train_out)

    test_in = tt.drop(["class"], axis=1).to_numpy()
    # test_in = np.array([[int(w.strip("b'")) for w in v] for v in test_in], dtype=np.int32)
    test_out = tt["class"].to_numpy()

    test = (test_in, test_out)

    inLayer = [len(tt.drop(["class"], axis=1).columns)]
    midLayers = [30,15,7,3]
    outLayer = [len(tt["class"].unique())]
    network = inLayer + midLayers + outLayer
    #pp.pprint(train)
    #pp.pprint(test)

    snn = Network(train, test, network)
    snn.Learn()
    snn.Test()
    snn.clean()

def s9():
    df = pd.read_csv("./datasets/data-1.csv")


    features = df.drop(["class"], axis=1).to_numpy()
    bl = len(features)
    
    features = np.array([[int(w.strip("b'")) for w in v] for v in features])
    F_max = np.max(features,axis=0)
    n_features = np.zeros(shape=(bl, sum(F_max)))
    F_max = np.cumsum(np.roll(np.append(F_max,[0]),1)[:len(F_max)])
    inds = features + F_max
    for i, v in zip(inds, n_features):
        for j in i:
            v[j-1] = 1
    
    out = df["class"].to_numpy()
    ndf = pd.DataFrame(n_features)
    ndf["class"] = out
    ndf.to_csv("./datasets/data-1-onehot.csv",index=None)



    
    