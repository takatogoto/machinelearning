import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump_sol import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of h_t for t=0,...,T-1
		self.betas = []       # list of beta_t for t=0,...,T-1
		self.errors = []
		return

	@abstractmethod

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO
		########################################################
		total_preds = np.zeros((len(features)))
		for idx in range(len(self.clfs_picked)):
			preds = self.clfs_picked[idx].predict(features)
			total_preds += np.inner(self.betas[idx], preds)
		total_preds = np.sign(total_preds).tolist()

		return total_preds


class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO
		############################################################
		# init. 
		N = len(features)
		w = 1/N * np.ones(N)
		
		# loop utill T
		for t in range(self.T):
			epsilon_t = 1e5
			self.errors.append([])
			# loop through all clfs
			for clf in self.clfs:
				pred = clf.predict(features)
				epsilon = np.sum(w*(pred != np.array(labels)))
				self.errors[t].append(epsilon)
				if epsilon < epsilon_t:
					epsilon_t = epsilon
					clf_t = clf
					pred_t = pred
			# update
			beta_t = 0.5 * np.log((1-epsilon_t)/epsilon_t)
			w *= (np.exp(-beta_t)*(np.array(labels)==pred_t) + np.exp(beta_t)*(np.array(labels)!=pred_t))
			w /= np.sum(w)
			self.clfs_picked.append(clf_t)
			self.betas.append(beta_t)
			# self.w = w
		return

		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)

	