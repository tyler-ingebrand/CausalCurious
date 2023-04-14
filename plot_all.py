import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import pickle

results_dir = "results"
alg = "td3_multi"
change_shape = True
change_size = False
change_mass = True
seeds = [0,1,2,3,4]
total_timesteps = 300000
random = 0.005
pickle_name = "data.pkl"
colors = ['r', 'b', 'g', 'orange', 'magenta']
for c, seed in zip(colors, seeds):
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
      dict = pickle.load(open(file, 'rb'))
      timesteps =  dict["timesteps"]
      mean_distance_my_cluster =  dict["mean_distance_my_cluster"]
      mean_distance_other_cluster =  dict["mean_distance_other_cluster"]
      success_rates = dict["success_rates"]
      averaging_step_size = len(success_rates)//3

      # plot
      # plt.plot(timesteps, success_rates, label="raw")
      success_rates = [np.mean(success_rates[max(i-averaging_step_size+1, 0):i+1]) for i in range(len(success_rates))]
      plt.plot(timesteps, success_rates, label=f"Seed {seed}")

      # plt.plot(timesteps, mean_distance_other_cluster, label=f"Seed {seed}", color=c)
      # plt.plot(timesteps, mean_distance_my_cluster, color=c)

plt.xlabel("Env Interactions")
plt.ylabel("Classification Success Rate (moving avg)")
plt.legend(loc="lower right")
plt.ylim(0.5, 1.05)
plt.title(f"{'Change Shape ' if change_shape else ''}{'Change Mass ' if change_mass else ''}{'Change Size ' if change_size else ''}")
plt.savefig(f"{results_dir}/{'change_shape_' if change_shape else ''}" 
            f"{'change_size_' if change_size else ''}"
            f"{'change_mass_' if change_mass else ''}"
            f"steps_{total_timesteps}_"
            f"random_{random}"
            f"_graph.png"
            )
plt.show()


