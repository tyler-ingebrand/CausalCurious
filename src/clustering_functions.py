import numpy as np
from matplotlib import pyplot as plt
from tslearn.clustering import TimeSeriesKMeans


def format_obs(obs, starts):
    # break up by number of trajectories for each env
    list_trajs = []
    for start, end in zip(starts[:-1], starts[1:]):
        list_trajs.append(obs[start:end])
    data = np.concatenate(list_trajs, axis=1)

    # transpose so tslearn is happy
    data = np.transpose(data, (1, 0, 2))
    return data

def cluster(data,
            n_clusters=2,
            distance_metric:str="softdtw",
            multi_process=True,
            plot=True,
            verbose=True):
    assert distance_metric == "softdtw" or distance_metric == "euclidean", "Distance metric must be one of 'euclidean' or 'softdtw', got {}".format(distance_metric)

    if verbose: print("Data: ", data.shape)
    if verbose: print("\t", data.shape[0], " trajectories of length ", data.shape[1], " with ", data.shape[2], " dimensions")

    # run clustering alg
    km = TimeSeriesKMeans(n_clusters=n_clusters,
                          metric=distance_metric,
                          max_iter=5,
                          max_iter_barycenter=5,
                          random_state=0,
                          n_jobs=-1 if multi_process else 0).fit(data)

    if verbose: print("Clusters: ", km.cluster_centers_.shape)
    if verbose: print("\t", km.cluster_centers_.shape[0], " clusters of length ", km.cluster_centers_.shape[1], " with ", km.cluster_centers_.shape[2], " dimensions")
    if verbose: print("Cluster labels: ", km.labels_)
    # if verbose: print(km.transform(data))
    # convert data for plotting
    if plot:
        for dimension in range(data.shape[2]):
            for cluster in range(km.cluster_centers_.shape[0]):
                for traj in range(data.shape[0]):
                    if km.labels_[traj] == cluster:
                        data_to_graph = data[traj, :, dimension].transpose()
                        plt.plot(data_to_graph, color='r' if cluster == 0 else 'b' if cluster == 1 else 'g', alpha=0.5,
                                 label='Trajectory in cluster {}'.format(cluster) if traj == 0 else None)

            # plot cluster centers
            clusters = km.cluster_centers_[:, :, dimension].transpose()
            plt.plot(clusters, color='black')[0].set_label('Cluster Center')

            # plot and show
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Dimension {} of data".format(dimension))
            plt.title("Example clustering usage")
            plt.show()
    return km