import pprint as pp
import numpy as np

# class Network():
#     def __init__(self, nShape):
#        self.weights = [np.random.randint(low=0, high=9, size=(nShape[i-1], nShape[i])) for i in range(1,len(nShape))]
#        self.bias = [np.zeros_like(x) for x in self.weights]

# SNN = Network([4,3,3,2])

# r = 10
# c = 4
# inp = np.random.random(size=(r, c))

# for i in range(3):
#     weights = SNN.weights[i].flatten()
#     bias = SNN.bias[i].flatten()
#     vals = []
#     nvals = []

#     if i == 0:
#         inpF = inp.flatten()
#         wCol = int(len(weights)/c)
#         for j in range(r):
#             for i in range(wCol):
#                 sum = 0
#                 for l in range(c):
#                     sum += inpF[j * c + l] * weights[l * wCol + i] + bias[l * wCol + i]

#                 nvals.append(float(sum))
#             vals.append(nvals)
#             nvals = []

            
#         print(np.array(vals))
                    

# print("dot")

# print(np.dot(inp, SNN.weights[0], ))

print(np.random.uniform(low=0,high=2/9,size=(3,4)))

