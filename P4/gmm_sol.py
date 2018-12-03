import numpy as np
from kmeans_sol import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception('Initialize variances, means, pi_k using k-means')
            k_means = KMeans(self.n_cluster, e=0.01)
            means, y, _ = k_means.fit(x)

            membership = np.zeros((N, self.n_cluster))
            membership[np.arange(N), y] = 1

            variances = np.zeros((self.n_cluster, D, D))
            for i in range(self.n_cluster):
                t = membership[:, i].reshape([-1, 1])
                variances[i] = (t * (x-means[i])).T @ (x -
                                                       means[i]) / (np.sum(t) + 1e-10)

            pi_k = np.sum(membership, axis=0)/N
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception('Initialize variances, means, pi_k randomly')
            means = np.random.rand(self.n_cluster, D)
            variances = np.zeros((self.n_cluster, D, D))
            for i in range(self.n_cluster):
                variances[i] = np.eye(D)
            pi_k = np.ones((self.n_cluster))/self.n_cluster
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement fit function (filename: gmm.py)')
        def compute_membership(self, x, means, variances, pi_k):
             gaussians = [self.Gaussian_pdf(means[i], variances[i])
                          for i in range(self.n_cluster)]
 
             N, D = x.shape
             membership = np.zeros((N, self.n_cluster))
             for i in range(N):
                 for j in range(self.n_cluster):
                     membership[i][j] = pi_k[j]*gaussians[j].getLikelihood(x[i])
             return membership/np.sum(membership, axis=1).reshape([-1, 1])
 
        l = self.compute_log_likelihood(x, means, variances, pi_k)
        #print('loglike', l)
 
        for j in range(self.max_iter):
            membership = compute_membership(self, x, means, variances, pi_k)

            # recompute mean
            for i in range(self.n_cluster):
                t = membership[:, i].reshape([-1, 1])
                means[i] = np.sum(t*x, axis=0) / (np.sum(t)+1e-10)
                variances[i] = (t * (x-means[i])).T @ (x -
                                                       means[i]) / (np.sum(t) + 1e-10)

            pi_k = np.sum(membership, axis=0)/N
            #print('variances', variances)
            l_new = self.compute_log_likelihood(x, means, variances, pi_k)
            if (np.abs(l_new-l) < self.e):
                self.means = means
                self.variances = variances
                self.pi_k = pi_k
                return j+1
            l = l_new
        self.means = means
        self.variances = variances
        self.pi_k = pi_k
        return self.max_iter

        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'

        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement sample function in gmm.py')

        D = self.means.shape[1]
        samples = np.random.standard_normal(size=(N, D))
        for i in range(N):
            component = np.random.choice(self.n_cluster, p=self.pi_k)
            samples[i] = np.random.multivariate_normal(mean=self.means[component], cov=self.variances[component])
        return samples

        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x, means=None, variances=None,pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None: 
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None: 
            pi_k = self.pi_k
                
        gaussians = [self.Gaussian_pdf(means[i], variances[i])
                     for i in range(self.n_cluster)]

        N, D = x.shape
        L = 0
        for i in range(N):
            p = 0
            for j in range(self.n_cluster):
                p = p + pi_k[j]*gaussians[j].getLikelihood(x[i])
            L = L + np.log(p)
        return float(L)

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None:  
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            D = variance.shape[0]
            flag = False
            while (np.linalg.matrix_rank(variance)!=D):
                variance = variance + np.eye(D) * 1e-3
            c = ((2*np.pi)**D)*np.linalg.det(variance)
            self.inv = np.linalg.inv(variance)
            self.c = c
            # DONOT MODIFY CODE BELOW THIS LINE
        
        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)')/sqrt(c)
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            likelihood = np.exp(-0.5 * np.dot(np.dot((x-self.mean),self.inv),(x-self.mean).T)) / np.sqrt(self.c)
            # DONOT MODIFY CODE BELOW THIS LINE
            return likelihood
