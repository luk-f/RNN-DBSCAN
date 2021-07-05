import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, euclidean

# enumerate rnn
def enum_rnn(X_indices, x_indice):
    return np.where(np.any(X_indices == x_indice, axis=1))

def density_cluster(cluster):
    return max(pdist(cluster))

class RNN_DBSCAN:

    def __init__(self, k, X):
        self.k = k
        self.X = X

        # voisinage 
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(self.X)
        self.indices = nbrs.kneighbors(self.X, return_distance=False)
        self.size_Rk = np.zeros(len(self.X), dtype=np.dtype('int32'))
        for i in range(len(self.X)):
            enum_rnn(self.indices, i)

        self.assign = np.zeros(len(self.X), dtype=np.dtype('int32'))

    def rnn_dbscan(self):
        """
        algo principal pour assigner une classe aux objets du dataset `X`
        `k` le paramètre de l'algo RNN-DBSCAN
        """

        self.assign = np.zeros(self.X.shape[0], dtype=np.dtype('int32'))
        cluster = 1
        for i, x in enumerate(self.X):
            if not self.assign[i]:
                if self.expand_cluster(x, i, cluster):
                    cluster += 1
        self.expand_clusters()
        return self.assign

    def expand_cluster(self, x, x_indice, cluster):
        
        if self.size_Rk[x_indice] < self.k:
            self.assign[x_indice] = -1
            return False
        else:
            # seeds = queue.Queue()
            # seeds.put(self.neighborhood(x, x_indice))
            seeds = self.neighborhood(x, x_indice)
            self.assign[x_indice] = cluster
            self.assign[seeds] = cluster
            for y in seeds:
                y_indice = np.unique(np.where(self.X == y)[0])[0]
                if self.size_Rk[y_indice] >= self.k:
                    for z in self.neighborhood(y, self.k):
                        z_assign = self.assign[np.unique(np.where(self.X == z)[0])]
                        # if z is unclassified
                        if z_assign == 0:
                            seeds = np.concatenate((seeds, [z]))
                            z_assign = cluster
                        # if z is noise
                        elif z_assign == -1:
                            z_assign = cluster
            return True

    def neighborhood(self, x, x_indice):
        """
        retourne l'ensemble des k voisins de `x` 
        et également ses voisins inverses qui sont hubs
        """
        x_indice = np.where(self.X == x)[0]
        rNN_x = enum_rnn(self.indices, x_indice)
        rNN_x = rNN_x[self.size_Rk[rNN_x] >= self.k]
        return np.concatenate((self.indices[x_indice], rNN_x), axis=None)

    def expand_clusters(self):
        for x in self.X:
            x_indice = np.unique(np.where(self.X == x)[0])
            if self.assign[x_indice][0] == -1:
                neighbors = self.indices[x_indice][0]
                mincluster = -1
                mindist = float('inf')
                for n in self.X[neighbors]:
                    n_indice = np.where(self.X == n)[0]
                    cluster = self.assign[n_indice[0]]
                    d = euclidean(x, n)
                    if self.size_Rk[n_indice[0]] >= self.k and d <= density_cluster(self.X[cluster]) and d < mindist:
                        mincluster = cluster
                        mindist = d
                self.assign[x_indice] = mincluster
    
if __name__ == "__main__":

    # paramètre
    k = 10
    # dataset
    dataset = np.concatenate((np.random.normal(loc=0.0, scale=1.0, size=(9_000, 2)), 
                              np.random.normal(loc=6.0, scale=1.0, size=(1_000, 2))), axis=0)
    my_clustering = RNN_DBSCAN(k, dataset)
    my_clustering.rnn_dbscan()