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
        num_cls = np.max(labels)+1 # number of class at the node

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

        #if node.splittable and not node.leaf:
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
        
        self.feature_remain = [] # delete feature list
        #if not hasattr(self, 'feature_remain'):
        #    self.feature_remain = [] # delete feature list
        #    print('first', self.feature_remain)
        
        #else:
        #    print('first2', self.feature_remain)
        #    self.feature_remain = list(set(self.feature_remain))
        #    print('next2', self.feature_remain)
        self.feature_remain = list(set(self.feature_remain))


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
            C, B = branp.shape
            # C axis = 0, B axis =1. C x B numpy array

            # number of element for each branch B x 1
            bele = np.sum(branp, axis = 0).reshape((B,1))
            
            # each class probability for each branch B x C
            Py_a = np.divide(np.transpose(branp), bele)
            
            # each branch probability B x 1
            Pa = bele / np.sum(branp)
            
            # entropy for each branch 1 x B
            H = np.sum(- np.multiply(Py_a, (np.ma.log2(Py_a)).filled(0)), axis=1)
            
            # conditional entropy
            coen = np.dot(H,Pa)[0].tolist()
            
            return coen
        
        
        if len(np.unique(self.labels)) == 1:
            self.splittable = False
            return
        
        elif not self.features[0]:
            self.splittable = False
            return
        
        elif len(np.unique(self.labels)) == 0:
            self.splittable = False
            return
        
        # handle as numpy
        feanp = np.array(self.features)
        labnp = np.array(self.labels)
        Ctemp = len(np.unique(self.labels))
        best_enropy = np.inf
        best_dim = 0
        
        #print('self.features')
        #print(self.features)
        #print('self.labels')
        #print(self.labels)
        
        #print('fea_remain', self.feature_remain)
        for idx_dim in range(len(self.features[0])):
            ############################################################
            # TODO: compare each split using conditional entropy
            #       find the best split
            ############################################################
            
            
            if idx_dim in self.feature_remain:
                continue
            # number of discrete value == number of branch
            # set() elminate duplication
            Bset = set([row[idx_dim] for row in self.features])
            Btemp = len(Bset)
            
            # initilize branch C x B array
            branchtm = np.zeros((Ctemp, Btemp))
            
            # loop for branch
            for i, valfea in enumerate(list(Bset)):
                #print("branch")
                #print(i)
                
                # loop for classes
                for j, clanm in enumerate(list(set(self.labels))):
                    #print("classes")
                    #print(j)
                    
                    # find number of ith branch(bool array for all sample)
                    branchbool = feanp[:,idx_dim] == valfea
                    # find number of jth classes(bool array for all sample)
                    classesbool = labnp == clanm
                    # find number of ith branch and jth classes
                    numberofbc = np.dot(1 * branchbool, 1* classesbool)
                    branchtm[j][i] = numberofbc

            branchlist = branchtm.astype(int).tolist()
            entemp = conditional_entropy(branchlist)
            #print(branchlist)

            #print('idx_dim: ', idx_dim)
            #print('branchlist: ', branchlist)
            #print('entropy: ', entemp)
            
            if entemp < best_enropy:
                best_enropy = entemp
                best_dim = idx_dim
                
        #print(best_dim)
        #print(best_enropy)
        #print('next')
        if best_enropy == np.inf:
            self.splittable = False
            return
        
        self.dim_split = best_dim
        self.feature_uniq_split = list(set([row[best_dim] for row in self.features]))
        #print('feature_uniq')
        #print(self.feature_uniq_split)


        ############################################################
        # TODO: split the node, add child nodes
        ############################################################

        #h_set.add(decision_stump.DecisionStump(s,b,d))
        for child in self.feature_uniq_split:
        #print('max feature_uniq_split', max(self.feature_uniq_split))
        #for child in range(max(self.feature_uniq_split)):
            ## add if empty situation
            
            # delete feature
            #childfea = np.delete(feanp[feanp[:, self.dim_split]==child], self.dim_split, axis=1).tolist()
            
            # no delete
            childfea = feanp[feanp[:, self.dim_split]==child].tolist()
            
            chilabel = labnp[feanp[:, self.dim_split]==child].tolist()
            #print('child',child)
            #print(chilabel)
            #print(childfea)
            #print(len(childfea[0]))
            #if len(chilabel) != 0:
               # print('uniquelabel')
               # print(np.unique(chilabel))
                #self.children.append(TreeNode(childfea, chilabel, len(np.unique(chilabel))))
                #self.children.append(TreeNode(childfea, chilabel, self.num_cls))
            #self.children.append(TreeNode(childfea, chilabel, self.cls_max))
            self.children.append(TreeNode(childfea, chilabel, self.num_cls))
            
            chil_idx = self.feature_uniq_split.index(child)
            self.children[chil_idx].feature_remain.extend(self.feature_remain)
            self.children[chil_idx].feature_remain.extend([self.dim_split])
            #if len(np.unique(chilabel)) == 0:
            #    print('chillabel',chilabel)
            #    self.children[child].cls_max = self.cls_max
            
        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable and (feature[self.dim_split] in self.feature_uniq_split):
        #if self.splittable:
            #print('feature', feature)
            #print('fea_uniq_split',self.feature_uniq_split)
            #print('dimp', self.dim_split)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max
