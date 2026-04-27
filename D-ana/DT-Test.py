import pandas as pd
import pprint as pp
import numpy as np

df = pd.read_csv("./datasets/data-1.csv")

class Node():
    def __init__(self, data, root=True, level=0, side=None, depth=None):
        self.level = level
        self.is_root = root
        self.data = data
        self.le_feature, self.le_val, self.is_leaf, self.le_class, self.le_feature_ind = self.find_vals(data)
        
        if not depth or not(depth == level):
            if  (not self.is_leaf):
                print("level | feature | feature value | is leaf | class | side")
                print(self.level, self.le_feature, self.le_val, self.is_leaf, self.le_class, side, sep=" | ")
                self.left_data = data[data[self.le_feature] == self.le_val]
                self.right_data = data[data[self.le_feature] != self.le_val]
                self.left = Node(self.left_data, root=False, level= self.level + 1, side="left")
                self.right = Node(self.right_data, root=False, level = self.level + 1, side="right")

            else:
                print(f"leaf at: {level}, {side} with {self.le_feature} == {self.le_val} and class {self.le_class}")

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

        print(f"min entropy: {int(min(e_vals))}")

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
    

dtree = Node(df)

#b0
print(dtree.query(["b'1'","b'1'","b'1'","b'1'","b'0'","b'2'"]))
