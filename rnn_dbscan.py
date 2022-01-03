import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, euclidean

from profiling import profile_decorator, ProfilingContext

# enumerate rnn
def enum_rnn(X_indices: np.ndarray, x_indice: int) -> np.array:
    rnn_indices = np.where(np.any(np.array(X_indices) == x_indice, axis=1))[0]
    return rnn_indices[rnn_indices != x_indice]

def rnn_size(rnn_indices):
    return np.bincount(rnn_indices.flatten(), minlength=rnn_indices.shape[0])

def density_cluster(cluster, metric = 'euclidean'):
    return max(pdist(cluster, metric = metric))

class RNN_DBSCAN:

    def __init__(self, X: np.ndarray, k: int = 10, metric = 'euclidean'):
        """
        Init RNN-DBSCAN

        :param X: les objets à clusteriser
        :type X: np.ndarray
        :param k: le paramètre de voisinage
        :type k: int
        """
        self.X = X
        self.k = k
        
        self.metric = metric
        if metric == 'euclidean':
            self.metric_function = euclidean
        else:
            self.metric_function = metric

        # voisinage 
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree', 
                                metric=metric).fit(self.X)
        self.indices = nbrs.kneighbors(self.X, return_distance=False)
        self.size_Rk = rnn_size(self.indices)

        self.assign = np.zeros(len(self.X), dtype=np.dtype('int32'))
        

    @profile_decorator
    def rnn_dbscan(self) -> np.array:
        """
        Algorithme principal pour assigner une classe aux objets du dataset `X`
        `k` le paramètre de l'algo RNN-DBSCAN

        :return: le groupe d'appartenance de chaque objet
        :rtype: np.array
        """

        self.assign = np.zeros(self.X.shape[0], dtype=np.dtype('int32'))
        cluster = 1
        for i, _ in enumerate(self.X):
            if not self.assign[i]:
                if self.expand_cluster(i, cluster):
                    cluster += 1
        self.expand_clusters()
        return self.assign

    @profile_decorator
    def expand_cluster(self, x_indice: int, cluster: int) -> bool:
        """
        Permet de créer un nouveau groupe autour de l'objet `x` si |Rk(x)| >= k
        Sinon retourne `False`

        :param x_indice: l'indice de l'objet `x` dans `self.X`
        :type x_indice: int
        :param cluster: le numéro de cluster
        :type cluster: int
        :return: retourne faux si l'objet `x` n'a pas un score de voisinage suffisant pour créer un groupe
        :rtype: bool
        """
        
        if self.size_Rk[x_indice] < self.k:
            self.assign[x_indice] = -1
            return False
        else:
            seeds = self.neighborhood(x_indice)
            self.assign[x_indice] = cluster
            self.assign[seeds] = cluster
            seeds_list = seeds.tolist()
            while len(seeds_list) > 0:
                y_indice = seeds_list.pop(0)
                if self.size_Rk[y_indice] >= self.k:
                    for z_indice in self.neighborhood(y_indice):
                        z_assign = self.assign[z_indice]
                        # if z is unclassified
                        if z_assign == 0:
                            if z_indice not in seeds_list:
                                seeds_list.append(z_indice)
                            self.assign[z_indice] = cluster
                        # if z is noise
                        elif z_assign == -1:
                            self.assign[z_indice] = cluster
            return True

    @profile_decorator
    def neighborhood(self, x_indice: int) -> np.array:
        """
        Retourne l'ensemble des k voisins de `x` 
        et également ses voisins inverses qui sont hubs (cad |Rk| >= k)

        :param x_indice: l'indice de l'objet `x`
        :type x_indice: int
        :return: les indices neighborhood
        :rtype: np.array
        """
        rNN_x = enum_rnn(self.indices, x_indice)
        if rNN_x.size == 0:
            return self.indices[x_indice]
        rNN_x = rNN_x[self.size_Rk[rNN_x] >= self.k]
        return np.unique(np.concatenate((self.indices[x_indice][1:], rNN_x), axis=None))

    @profile_decorator
    def expand_clusters(self):
        """
        Étend les groupes.
        Pour les objets `x` classées comme anormaux : on regarde ses knn.
        Sous certaines conditions (un voisin hub, ...), `x` peut finalement appartenir à un groupe.
        
        """
        # TODO for x_indice, x in enumerate(self.X[self.assign == -1]):
        for x_indice, x in enumerate(self.X):
            if self.assign[x_indice] == -1:
                neighbors = self.indices[x_indice]
                mincluster = -1
                mindist = float('inf')
                for n_indice in neighbors:
                    n = self.X[n_indice]
                    cluster_num = self.assign[n_indice]
                    cluster = self.X[np.where(self.assign == cluster_num)]
                    d = euclidean(x, n)
                    if self.size_Rk[n_indice] >= self.k and d <= density_cluster(cluster, metric=self.metric) and d < mindist:
                        mincluster = cluster_num
                        mindist = d
                self.assign[x_indice] = mincluster
    
