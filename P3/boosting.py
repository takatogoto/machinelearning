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
        Dn = np.zeros((self.T + 1, N))
        
        # 1
        Dn[0,:] = 1/N
        
        # 2
        for t in range(self.T):
            
            # 3
            htemp = []
            eps = np.inf
            beta = 0.0
            for cl in self.clfs:
                #print(cl.predict(features))
                findht = np.multiply(Dn[t,:], 
                                     (np.array(labels) != np.array(cl.predict(features))))
                #print(cl.predict(features))
                #print(np.array(labels) != np.array(cl.predict(features)))
                
                if np.sum(findht) < eps:
                    htemp = cl
                    eps = np.sum(findht) # 4
            
            self.clfs_picked.extend([htemp])
            
            # 5
            beta = 0.5 * np.log((1 - eps) / eps)
            self.betas.extend([beta])
            
            # 6
            Dn[t+1, :] = np.exp(-beta) * np.multiply(Dn[t, :], (np.array(labels)
                                                                == np.array(self.clfs_picked[t].predict(features))))\
            + np.exp(beta) * np.multiply(Dn[t, :], (np.array(labels)
                                                                != np.array(self.clfs_picked[t].predict(features))))
            Dn[t+1, :] = Dn[t+1, :] / np.sum(Dn[t+1, :])
            #print(Dn)
            print(self.betas)
                
            
    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)



    