import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import pickle

results_dir = "results"
alg = "td3"
change_shape = False
change_size = False
change_mass = True
seed = 1
total_timesteps = 300000
pickle_name = "data.pkl"

# find file location
s1 = "change_shape_" if change_shape else ""
s2 = "change_size_" if change_size else ""
s3 = "change_mass_" if change_mass else ""
dir = f"{results_dir}/{alg}_{s1}{s2}{s3}seed_{seed}_steps_{total_timesteps}"
file = f"{dir}/{pickle_name}"

# open file
dict = pickle.load(open(file, 'rb'))
timesteps =  dict["timesteps"]
mean_distance_my_cluster =  dict["mean_distance_my_cluster"]
mean_distance_other_cluster =  dict["mean_distance_other_cluster"]
success_rates = dict["success_rates"]
averaging_step_size = len(success_rates)//3

# plot
plt.plot(timesteps, success_rates, label="raw")
success_rates = [np.mean(success_rates[max(i-averaging_step_size+1, 0):i+1]) for i in range(len(success_rates))]
plt.plot(timesteps, success_rates, label="averaged")
plt.legend()
plt.show()


