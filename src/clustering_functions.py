import os

import numpy as np
from matplotlib import pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_soft_dtw, soft_dtw_alignment


def format_obs(obs, starts, normalize=True):
    # break up by number of trajectories for each env
    list_trajs = []
    for start, end in zip(starts[:-1], starts[1:]):
        list_trajs.append(obs[start:end])
    data = np.concatenate(list_trajs, axis=1)

    # transpose so tslearn is happy. Is in n_trajs X n_timesteps X state space dimensions
    data = np.transpose(data, (1, 0, 2))

    # normalize all data to be 0 mean and 1 std dev
    if normalize:
        for dimension in range(data.shape[2]):
            slice = data[:, :, dimension]
            mean = np.mean(slice)
            std = np.std(slice)
            data[:, :, dimension] = (data[:, :, dimension] - mean)/std



    return data

def cluster(data,
            n_clusters=2,
            distance_metric:str="softdtw",
            multi_process=True,
            plot=True,
            verbose=True,
            timestep=0,
            debug_dir=None):
    assert distance_metric == "softdtw" or distance_metric == "euclidean", "Distance metric must be one of 'euclidean' or 'softdtw', got {}".format(distance_metric)
    plt.rcParams["figure.figsize"] = (3.5*5, 3.5*4)
    if verbose:
        print("\t", data.shape[0], " trajectories of length ", data.shape[1], " with ", data.shape[2], " dimensions")

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
        dimension_names = {0: "X",
                           1: "Y",
                           2: "Z",
                           3: "Rotation X",
                           4: "Rotation Y",
                           5: "Rotation Z",
                           6: "Rotation W",
                           7: "Vel. X",
                           8: "Vel. Y",
                           9: "Vel. Z",
                           10: "Ang. Vel. X",
                           11: "Ang. Vel. Y",
                           12: "Ang. Vel. Z",
                           }
        for dimension in range(data.shape[2]):
            labeled = [False for c in range(km.cluster_centers_.shape[0])]
            plt.subplot(4, 5, dimension + 1)
            for cluster in range(km.cluster_centers_.shape[0]):
                for traj in range(data.shape[0]):
                    if km.labels_[traj] == cluster:

                        data_to_graph = data[traj, :, dimension].transpose()
                        plt.plot(data_to_graph, color='r' if cluster == 0 else 'b' if cluster == 1 else 'g', alpha=0.5,
                                 label='Trajectory in cluster {}'.format(cluster) if not labeled[cluster] else None)
                        if not labeled[cluster]:
                            labeled[cluster] = True


            # plot cluster centers
            clusters = km.cluster_centers_[:, :, dimension].transpose()
            plt.plot(clusters, color='black')[0].set_label('Cluster Center')

            # plot and show
            plt.title(dimension_names[dimension])
            if dimension == 2:
                plt.legend(loc='upper center')
        plt.tight_layout()

        # create diretories for results
        if not debug_dir:
            os.makedirs("./debug/cluster_plots/", exist_ok=True)
            plt.savefig("./debug/cluster_plots/timestep_{}.png".format(timestep))
        else:
            os.makedirs("{}/debug/cluster_plots/".format(debug_dir), exist_ok=True)
            plt.savefig("{}/debug/cluster_plots/timestep_{}.png".format(debug_dir, timestep))
        plt.clf()
        return km

def compute_distance_between_trajectory_and_cluster(traj, cluster):
    # computes pairwise distance between all timesteps in traj and cluster
    distances = np.zeros((traj.shape[0], cluster.shape[0]))
    for traj_timestep in range(traj.shape[0]):
        for cluster_timestep in range(cluster.shape[0]):
            distances[traj_timestep, cluster_timestep] = np.linalg.norm(traj[traj_timestep] - cluster[cluster_timestep])

    # computes allignment matrix for all pairs of points between traj and cluster.
    # 1 = well alligned, 0 not alligned. Always between 0 and 1
    alignment, sim = soft_dtw_alignment(traj, cluster, gamma=1.0)

    # element wise multiplication. Distance depends on if it is well aligned or not.
    result = distances * alignment

    # sum distances along the trajectory. Gets distance from cluster at every timestep
    summed_result = np.sum(result, axis=1)
    return summed_result

def get_distances_between_trajectories_and_clusters(cluster_labels, cluster_centers, data, verbose=False, plot=True):
    # compute distances between current cluster and other cluster
    distance_to_my_cluster = np.zeros((data.shape[0], data.shape[1]))
    distance_to_other_cluster = np.zeros((data.shape[0], data.shape[1]))

    for index, traj in enumerate(data):
        if verbose: print(index)
        label = cluster_labels[index]
        my_cluster = cluster_centers[label]
        other_cluster = cluster_centers[1 - label]
        dist_mine = compute_distance_between_trajectory_and_cluster(traj, my_cluster)
        dist_other = compute_distance_between_trajectory_and_cluster(traj, other_cluster)
        distance_to_my_cluster[index] = dist_mine
        distance_to_other_cluster[index] = dist_other

    if plot:
        plt.plot(np.transpose(distance_to_my_cluster), color='b')[0].set_label("Distance to my cluster")
        plt.plot(np.transpose(distance_to_other_cluster), color='r')[0].set_label("Distance to other cluster")
        plt.legend()
        plt.show()

    return distance_to_my_cluster, distance_to_other_cluster

def normalize_distances(ts_dists):
    '''
    Normalizes the distances between time series and a cluster. 
        Should be consistent with type of cluster, ie. distance should either be between all trajectories  
        and their own cluster or distance between all  trajectories  and other cluster
    Inuput
        ts_dists: matrix of all time series distance to their own clusters, shape: (num_trajectories x N)
    
    Return: Normalized ts_distances. This should be shape (num_trajectories x N), N = timesteps
    '''
    max_ts_dist = np.max(ts_dists)
    min_ts_dist = np.min(ts_dists)

    return (ts_dists - min_ts_dist) / (max_ts_dist - min_ts_dist)


def get_reward(norm_dist_own, norm_dist_other):
    '''
    Calculate reward Encouraging distance from other cluster and proximity to own cluster
    N = num time steps
    Inputs:
    norm_dist_own: (numpy array, shape: num_trajectories x N) normalized time warped distance from trajectories to their own cluster centers 
    norm_dist_other: (numpy array, shape: num_trajectories x N) normalized time warped distance from trajectories to the other cluster centers
    '''

    return norm_dist_other - norm_dist_own 




def get_change_in_distance(distance_to_my_cluster, distance_to_other_cluster):
    distance_to_my_cluster[:, :-1] = distance_to_my_cluster[:, 1:] - distance_to_my_cluster[:, :-1]
    distance_to_other_cluster[:, :-1] = distance_to_other_cluster[:, 1:] - distance_to_other_cluster[:, :-1]

    distance_to_my_cluster[:, -1] = 0.0
    distance_to_other_cluster[:, -1] = 0.0


    return distance_to_my_cluster, distance_to_other_cluster
