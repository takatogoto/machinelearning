import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
    def __init__(self, s:int, b:float, d:int):
        self.clf_name = "Decision_stump"
        self.s = s
        self.b = b
        self.d = d

    def train(self, features: List[List[float]], labels: List[int]):
        pass
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
        Inputs:
        - features: the features of all test examples
   
        Returns:
        - the prediction (-1 or +1) for each example (in a list)
        '''
        ##################################################
        # TODO: implement "predict"
        ##################################################
        
        #h = np.zeros(len(features))
        #for num in range(len(features)):
        #    if features[num][d] > b:
        #        h[num] = s
        #    else:
        #        h[num] = -s
        #predict = h.tolist()
        hbool = np.array(features)[:,self.d] > self.b
        predict = ((hbool * 2 - 1) * self.s).tolist()
        
        
        return predict
        
        