import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import pickle

results_dir = "results"
alg = "td3"
change_shape = True
change_size = False
change_mass = False
seeds = [0, 1, 2, 3, 4]
total_timesteps = 300000
randoms = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
colors = matplotlib.colormaps['plasma'].resampled(len(randoms)).colors
pickle_name = "data.pkl"

for random, color in zip(randoms, colors):
    s_rates_total = None
    for seed in seeds:
        # find file location
        dir = f"{results_dir}/{alg}" \
              f"_{'change_shape_' if change_shape else ''}" \
              f"{'change_size_' if change_size else ''}" \
              f"{'change_mass_' if change_mass else ''}" \
              f"seed_{seed}_" \
              f"steps_{total_timesteps}_" \
              f"random_{random}"
        file = f"{dir}/{pickle_name}"

        # open file
        try:
            dict = pickle.load(open(file, 'rb'))
        except:
            continue
        timesteps = dict["timesteps"]
        mean_distance_my_cluster = dict["mean_distance_my_cluster"]
        mean_distance_other_cluster = dict["mean_distance_other_cluster"]
        success_rates = dict["success_rates"]
        averaging_step_size = len(success_rates) // 3

        # plot
        # plt.plot(timesteps, success_rates, label="raw")
        # success_rates = [np.mean(success_rates[max(i - averaging_step_size + 1, 0):i + 1]) for i in range(len(success_rates))]
        success_rates = numpy.array(success_rates)
        s_rates_total = success_rates if s_rates_total is None else s_rates_total + success_rates
    if s_rates_total is None:
        continue
    s_rates_total = numpy.array([np.mean(s_rates_total[max(i - averaging_step_size + 1, 0):i + 1]) for i in range(len(s_rates_total))])
    plt.plot(timesteps, s_rates_total/len(seeds), label=f"{random}",  color=color)
plt.xlabel("Env Interactions")
plt.ylabel("Classification Success Rate (moving avg)")
plt.legend(loc="upper left")
plt.title("Initial State Randomness")
plt.ylim(0.5, 1.0)
plt.savefig(f"{results_dir}/{'change_shape_' if change_shape else ''}"
            f"{'change_size_' if change_size else ''}"
            f"{'change_mass_' if change_mass else ''}"
            f"steps_{total_timesteps}_"
            f"random_sweep"
            f"_graph.png"
            )
plt.show()
