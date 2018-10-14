import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs      # set of weak classifiers to be considered
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T
    
        self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
        self.betas = []       # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:
        '''
        Inputs:
        - features: the features of all test examples
   
        Returns:
        - the prediction (-1 or +1) for each example (in a list)
        '''
        ########################################################
        # TODO: implement "predict"
        ########################################################
        
        bh = np.zeros((self.T, len(features)))
        #for t in range(self.T):
        #    bh[t,:] = self.betas[t] * np.array(self.clfs_picked[t].predict(features)) ##clfs argv
        t = 0
        for clfset in self.clfs_picked:
            bh[t,:] = self.betas[t] * np.array(clfset.predict(features))
            t += 1
        
        predict = (np.sum(bh,axis=0)).astype(int).tolist()
        
        return predict
        

class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return
        
    def train(self, features: List[List[float]], labels: List[int]):
        '''
        Inputs:
        - features: the features of all examples
        - labels: the label of all examples
   
        Require:
        - store what you learn in self.clfs_picked and self.betas
        '''
        ############################################################
        # TODO: implement "train"
        ############################################################
        N = len(features)
        D = np.zeros(self.T + 1, N)
        D[0,:] = 1/N
        for t in range(self.T):
            self.clfs_picked.extend(clfs
        
        
        
        
    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)



    