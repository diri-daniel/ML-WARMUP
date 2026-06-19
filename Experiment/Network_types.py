from enum import StrEnum

class NetworkType(StrEnum):
    Simple_Neural_Network = "SNN"
    K_Nearest_Neighbours = "KNN"
    Decision_Tree = "DT"

    @property
    def display_name(self):
        return self.name.replace("_", " ")