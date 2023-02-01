import numpy as np
from matplotlib import pyplot as plt
from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_soft_dtw, soft_dtw_alignment

# random data
X = random_walks(n_ts=50, sz=32, d=1)
print("Data: ", X.shape)
print("\t", X.shape[0], " trajectories of length ", X.shape[1], " with ", X.shape[2], " dimensions")

# run clustering alg
km = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5, max_iter_barycenter=5, random_state=0, n_jobs=-1).fit(X)
# km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, random_state=0).fit(X)
print("Clusters: ", km.cluster_centers_.shape)
print("\t", km.cluster_centers_.shape[0], " clusters of length ", km.cluster_centers_.shape[1], " with ", km.cluster_centers_.shape[2], " dimensions")
print("Cluster labels: ", km.labels_)
print("\n\n\n")
# print(km.transform(X))


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


traj0 = X[0, :, :]
cluster = km.cluster_centers_[km.labels_[0]]
distances = cdist_soft_dtw(traj0, cluster, gamma=1.0)
print("cdist: ", distances.shape)
print(distances)

alignment, sim = soft_dtw_alignment(traj0, cluster, gamma=1.0)
print("align:", alignment.shape)
print(alignment)


# element wise multiplication
result = distances * alignment
print(result)

summed_result = np.sum(result, axis=1)
print(summed_result)


plt.cla()
plt.plot(cluster, color='black')
plt.plot(traj0, color='b')
for timestep, value in enumerate(traj0):
    best_alligment = np.max(alignment, axis=)
    plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
plt.show()

plt.plot(summed_result)
plt.show()