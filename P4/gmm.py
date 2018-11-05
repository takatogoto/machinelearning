import numpy as np
from kmeans import KMeans

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
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
            #    'Implement initialization of variances, means, pi_k using k-means')
            
            kmean = KMeans(self.n_cluster, self.max_iter, self.e) # self.n_cluster self.max_iter 
            self.means, ymu, _ = kmean.fit(x) # self.means
            self.pi_k = np.array([np.sum(ymu == k) for k in range(self.n_cluster)]) / N #self.pi_k self.n_cluster
            
            # gamma_ik = {0, 1} at this initialize
            self.variances = np.zeros((self.n_cluster, D, D)) #self.variances self.n_cluster self.means
            for k in range(self.n_cluster):
                xt = x[ymu == k,:] - self.means[k,:] 
                self.variances[k, :, :] = np.dot(np.transpose(xt),xt) / np.sum(ymu==k) #self.variances 
                       
            
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
            #    'Implement initialization of variances, means, pi_k randomly')
            
            self.means = np.random.rand(self.n_cluster, D) # self.means self.n_cluster
            self.pi_k = np.random.rand(self.n_cluster,) #self.pi_k self.n_cluster
            self.variances = np.random.rand(self.n_cluster, D, D) #self.variances self.n_cluster

            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement fit function (filename: gmm.py)')
        
        #4
        l = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
        gamma = np.zeros((N, self.n_cluster))
        for itr in range(self.max_iter):
            #print("Iteration,", itr)
            #print('variances', self.variances[0,:,:])
            #print('means', self.means[0,:] )
            # 6 E step
            for n in range(N):
                sumnorm = .0
                normk = np.zeros(self.n_cluster)
                for k in range(self.n_cluster):
                    gaus = self.Gaussian_pdf(
                        self.means[k,:], self.variances[k,:]).getLikelihood(x[n,:])
                    #if gaus ==0:
                    #    print('gaus is 0 when n, k', n , k)
                    normk[k]= self.pi_k[k] * gaus
                    if np.isnan(normk[k]):
                        normk[k] =0  
                    sumnorm += normk[k]
                #gamma[n, :] = normk / sumnorm
                if sumnorm==0:
                    gamma[n, :] = 1/self.n_cluster
                else:
                    gamma[n, :] = normk / sumnorm
            #print('min gamma', np.min(gamma))
            #print('gamma',gamma)
            #print('gamma -3 :', gamma[-3,:])
            #print(np.isnan(gamma[-3,0]))
            #print('Estep')
            # 7 M step
            # eq.(5)
            Nk = np.sum(gamma, axis=0)
            #print('Nk',Nk)
            # eq.(6)
            means2 = np.zeros(self.means.shape)
            for k in range(self.n_cluster):
                means2[k, :] = np.sum(np.multiply(gamma[:, k].reshape(gamma.shape[0],1), x), axis=0)/Nk[k]
            
            #print('means', means2[0,:])
            #print('Eq6')
            # eq.(7)
            variances2 = np.zeros(self.variances.shape)
            for k in range(self.n_cluster):
                sumvark = np.zeros((D,D))
                for n in range(N):
                    xmu = x[n, :] - self.means[k, :]
                    sumvark += gamma[n, k]*(np.dot(np.transpose(xmu), xmu))
                variances2[k, :, :] = sumvark / Nk[k]
            #print(variances2[0, :, :])
            
            #print('Eq7')
            # eq.(8)
            self.pi_k = Nk / N
            self.means = means2
            self.variances = variances2
            #print('variances', self.variances)
            #print('Eq8')
            l1 = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
            
            # stop condition
            if abs(l-l1) < self.e:
                break
            l = l1
            
        return itr
        
        
        
        # DONOT MODIFY CODE BELOW THIS LINE


    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement sample function in gmm.py')
        
        samples = np.zeros((N, self.means.shape[1]))
        firstsa = np.random.choice(self.n_cluster, N, p=self.pi_k)
        for n in range(N):
            samples[n] = np.random.multivariate_normal(
                self.means[firstsa[n]], 
                self.variances[firstsa[n]])
            
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
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
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement compute_log_likelihood function in gmm.py')
        
        N, D = x.shape
        K = self.pi_k.shape[0]
        '''
        log_likelihood = .0
        for n in range(N):
            lnk = .0
            #print(lnk)
            for k in range(K):
                lnk += self.pi_k[k] * self.Gaussian_pdf(
                    self.means[k,:], self.variances[k,:]).getLikelihood(x[n,:])
            #print(lnk)
            log_likelihood += np.log(lnk)
        '''
        log_likelihood = sum([np.log
                              (sum(
                                  [self.pi_k[k] * self.Gaussian_pdf(self.means[k,:], self.variances[k,:]).getLikelihood(x[n,:]) 
                                   for k in range(K)])) for n in range(N)])
        log_likelihood = log_likelihood.tolist()

        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

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
            #raise Exception('Impliment Guassian_pdf __init__')
            
            D = self.variance.shape[1] # self.variance
            while np.linalg.matrix_rank(self.variance) != len(self.variance): # self.variance
                #print('inverse')
                #print(self.variance)
                self.variance += 1e-3 * np.identity(len(self.variance)) # self.variance
            self.inv = np.linalg.inv(self.variance) # self.variance self.inv
            self.c = np.abs(((2*np.pi)**D) * np.linalg.det(self.variance)) # self.c self.variance            

            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)') / sqrt(c)
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception('Impliment Guassian_pdf getLikelihood')
            
            p = np.exp(-0.5 * np.dot(np.dot((x - self.mean), self.inv),
                                     np.transpose(x - self.mean))) / np.sqrt(self.c) # self.mean self.inv self.c
            if np.abs(p) < 1e-200:
                p = 1e-200

            # DONOT MODIFY CODE BELOW THIS LINE
            return p
