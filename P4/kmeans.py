import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        ''' 
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement fit function in KMeans class (filename: kmeans.py)')
        
        # initialize mu and J
        mu0_idx = np.random.choice(range(N), self.n_cluster)
        mu = x[mu0_idx,:]
        j = 1e10

        def compute_ridx(x, mu):
            # input (N X D) ,(C X D)
            # output (C X 1)
            A = np.sum(x**2,axis=1).reshape(x.shape[0],1)
            B = np.sum(mu**2,axis=1).reshape(mu.shape[0],1)
            AB = np.dot(x, np.transpose(mu))
            dists = np.sqrt(-2*AB + A + np.transpose(B))
    
            # because r is {0,1}, just return index 
            r = np.argmin(np.transpose(dists), axis=1)
            return np.ravel(r)



        for itr in range(self.max_iter):    
    
            r_idx = compute_ridx(mu, x)
    
            # r_ik = 0 if xi not in k
            # mu[r_idx,:] means mu_k when r_ik = 1, size N X D
            #j1 = sum(np.sum(np.multiply((x - mu[r_idx,:]), (x - mu[r_idx,:])), axis=1))
            j2 = np.sum(np.multiply((mu[r_idx,:] - x ), (mu[r_idx,:] - x)))
            #print(j-j2)
            
            # stop condition
            if abs(j-j2) < self.e:
                break
            
            j = j2
            # r_idx is asigned index for each class
            mu = np.array(
                [np.average(x[r_idx==i,:], axis=0) for i in range(self.n_cluster)])
            
        y = compute_ridx(mu, x)
        return (mu, y, itr)        
        
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement fit function in KMeansClassifier class (filename: kmeans.py)')
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e) ## self.n ....
        centroids, membership, i = kmeans.fit(x)
        
        centroid_labels = np.zeros((self.n_cluster,)) # self.n_cluster
        for k in range(self.n_cluster): # self.n_cluster
            centroid_labels[k] = np.argmax(np.bincount(y[membership==k]))        
        
        self.centroids = centroids
        self.centroid_labels = centroid_labels
        # DONOT CHANGE CODE BELOW THIS LINE


        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement predict function in KMeansClassifier class (filename: kmeans.py)')
        
        def compute_ridx(x, mu):
            # input (N X D) ,(C X D)
            # output (C X 1)
            A = np.sum(x**2,axis=1).reshape(x.shape[0],1)
            B = np.sum(mu**2,axis=1).reshape(mu.shape[0],1)
            AB = np.dot(x, np.transpose(mu))
            dists = np.sqrt(-2*AB + A + np.transpose(B))
    
            # because r is {0,1}, just return index 
            r = np.argmin(np.transpose(dists), axis=1)
            return np.ravel(r)
        
        mincent = compute_ridx(self.centroids, x) # self.centroids
        labels = self.centroid_labels[list(mincent)] # self.centroids_labels
        
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

