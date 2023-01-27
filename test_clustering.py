from matplotlib import pyplot as plt
from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans

# random data, create clusters
X = random_walks(n_ts=50, sz=32, d=1)
print("Data: ", X.shape)
print("\t", X.shape[0], " trajectories of length ", X.shape[1], " with ", X.shape[2], " dimensions")
# km = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5, max_iter_barycenter=5, random_state=0).fit(X)
km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, random_state=0).fit(X)
print("Clusters: ", km.cluster_centers_.shape)
print("\t", km.cluster_centers_.shape[0], " clusters of length ", km.cluster_centers_.shape[1], " with ", km.cluster_centers_.shape[2], " dimensions")
print("Cluster labels: ", km.labels_)

# convert data for plotting
for dimension in range(X.shape[2]):
    for cluster in range(km.cluster_centers_.shape[0]):
        data = X[km.labels_==cluster,:,dimension].transpose()
        plt.plot(data, color='r' if cluster == 0 else 'g' if cluster == 1 else 'b', alpha=0.5)[0].set_label('Trajectory in cluster {}'.format(cluster))

    # plot cluster centers
    clusters = km.cluster_centers_[:,:,dimension].transpose()
    plt.plot(clusters, color='black')[0].set_label('Cluster Center')

    # plot and show
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Dimension {} of data".format(dimension))
    plt.title("Example clustering usage")
    plt.show()