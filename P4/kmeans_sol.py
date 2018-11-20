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
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
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
        # raise Exception(
        #     'Implement fit function in KMeans class (filename: kmeans.py)')
        def __assign_cluster(self, x, centroids):
            N = x.shape[0]
            distances = np.zeros((N, self.n_cluster))

            for i in range(self.n_cluster):
                distances[:, i] = np.sum((x-centroids[i])**2, axis=1)

            y = np.argmin(distances, axis=1)
            return y

        def __compute_centroids(self, x, y):
            d = x.shape[1]
            centroids = np.zeros((self.n_cluster, d))

            for i in range(self.n_cluster):
                membership = (y == i).reshape([-1, 1])
                centroids[i] = np.sum(membership*x, axis=0) / \
                    (np.sum(membership)+1e-10)

            return centroids

        def __compute_distortion(self, x, centroids):
            N = x.shape[0]
            distance = np.zeros((N, self.n_cluster))

            for i in range(self.n_cluster):
                distance[:, i] = np.sum((x-centroids[i])**2, axis=1)
            return np.sum(np.min(distance, axis=1))

        idx = np.random.choice(N, size=self.n_cluster)
        centroids = x[idx]
        J = 1e10
        for i in range(self.max_iter):
            y = __assign_cluster(self, x, centroids)
            J_new = __compute_distortion(self, x, centroids)
            if (np.abs(J-J_new)/N < self.e):
                return centroids, y, i+1

            centroids = __compute_centroids(self, x, y)
            J = J_new
        return centroids, y, self.max_iter
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
                    majority voting (N,) numpy array) 
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
        # raise Exception(
        #     'Implement fit function in KMeansClassifier class (filename: kmeans.py')

        k_means = KMeans(self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, _ = k_means.fit(x)

        centroid_labels = []
        for i in range(self.n_cluster):
            y_ = y[(membership == i)]
            if (y_.size > 0):
                _, idx, counts = np.unique(
                    y_, return_index=True, return_counts=True)

                index = idx[np.argmax(counts)]
                centroid_labels.append(y_[index])
            else:
                centroid_labels.append(0)
        centroid_labels = np.array(centroid_labels)
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

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
        # raise Exception(
        #     'Implement predict function in KMeansClassifier class (filename: kmeans.py')
        n_cluster = self.centroids.shape[0]
        N = x.shape[0]
        distances = np.zeros((N, n_cluster))

        for i in range(n_cluster):
            distances[:, i] = np.sum((x-self.centroids[i])**2, axis=1)
        centroid_idx = np.argmin(distances, axis=1)
        labels = []
        for i in centroid_idx:
            labels.append(self.centroid_labels[i])
        return np.array(labels)
        # DONOT CHANGE CODE BELOW THIS LINE
