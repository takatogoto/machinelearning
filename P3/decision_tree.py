import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert(len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels)+1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return
        
    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')
        
        string = ''
        for idx_cls in range(node.num_cls):
            string += str(node.labels.count(idx_cls)) + ' '
        print(indent + ' num of sample / cls: ' + string)

        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent+'}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None # the index of the feature to be split

        self.feature_uniq_split = None # the possible unique values of the feature to be split


    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
            branches: C x B array, 
                      C is the number of classes,
                      B is the number of branches
                      it stores the number of 
                      corresponding training samples 
                      e.g.
                                  ○ ○ ○ ○
                                  ● ● ● ●
                                ┏━━━━┻━━━━┓
                               ○ ○       ○ ○
                               ● ● ● ●
                               
                      branches = [[2,2], [4,0]]
            '''
            ########################################################
            # TODO: compute the conditional entropy
            ########################################################
            
            branp = np.array(branches)
            B, C = branp.shape
            # B axis = 0, C axis =1. B x C numpy array

            # number of element for each branch B x 1
            bele = np.sum(branp, axis = 1).reshape((B,1))
            
            # each class probability for each branch B x C
            Py_a = np.divide(branp, bele)
            
            # each branch probability B x 1
            Pa = bele / np.sum(branp)
            
            # entropy for each branch 1 x B
            H = np.sum(- np.multiply(Py_a, (np.ma.log2(Py_a)).filled(0)), axis=1)
            
            # conditional entropy
            coen = np.dot(H,Pa)[0].tolist()
            
            return coen
        
        
        # handle as numpy
        feanp = np.array(self.features)
        labnp = np.array(self.labels)
        Ctemp = len(np.unique(self.labels))
        best_enropy = np.inf
        best_dim = 0
        
        for idx_dim in range(len(self.features[0])):
            ############################################################
            # TODO: compare each split using conditional entropy
            #       find the best split
            ############################################################
            
            # number of discrete value == number of branch
            # set() elminate duplication
            Bset = set([row[idx_dim] for row in self.features])
            Btemp = len(Bset)
            
            # initilize branch C x B array
            branchtm = np.zeros((Ctemp, Btemp))
            
            # loop for branch
            for i, valfea in enumerate(list(Bset)):
                print("branch")
                print(i)
                
                # loop for classes
                for j, clanm in enumerate(list(set(labels))):
                    print("classes")
                    print(j)
                    
                    # find number of ith branch(bool array for all sample)
                    branchbool = feanp[:,idx_dim] == valfea
                    # find number of jth classes(bool array for all sample)
                    classesbool = labnp == clanm
                    # find number of ith branch and jth classes
                    numberofbc = np.dot(branchbool, classesbool)
                    branchtm[j][i] = numberofbc

            branchlist = branchtm.astype(int).tolist()
            entemp = conditional_entropy(branchlist)
            
            if entemp < best_enropy:
                best_enropy = entemp
                best_dim = idx_dim
                
            print(best_dim)
            print(best_enropy)           
            self.dim_split = best_dim
            self.feature_uniq_split = [row[best_dim] for row in self.features]


        ############################################################
        # TODO: split the node, add child nodes
        ############################################################




        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            # print(feature)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max
