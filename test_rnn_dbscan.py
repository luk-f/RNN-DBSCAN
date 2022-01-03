from rnn_dbscan import RNN_DBSCAN
from pyhaversine import haversine

import numpy as np
from profiling import ProfilingContext

# paramètre
k = 2
# dataset
dataset = np.concatenate((np.random.normal(loc=0.0, scale=1.0, size=(9_000, 2)), 
                            np.random.normal(loc=6.0, scale=1.0, size=(1_000, 2))), axis=0)

my_clustering = RNN_DBSCAN(X=dataset, k=k, metric=haversine)

labels = my_clustering.rnn_dbscan()

print(np.unique(labels, return_counts =True))

print("---")
ProfilingContext.print_summary()

