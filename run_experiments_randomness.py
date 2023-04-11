import os
from multiprocessing import Pool, freeze_support
import time
from test_td3 import test
import subprocess

if __name__ == "__main__":
    freeze_support()

    n_envs = 32
    seeds = 5
    number_steps= 300_000
    bools = {"change_shape": True, "change_mass": False, "change_size": False}
    randoms = [0.0001, 0.0005,  0.001, 0.01, 0.05,  0.1]
    multi_process = False
    max_processes_at_once = 1
    os.makedirs("results", exist_ok=True)

    if not multi_process:
        for randomness in randoms:
             for s in range(seeds):
                 subprocess.run(
                                f"..\\venv\\Scripts\\python.exe ./test_td3.py "
                                f"--seed {s} "
                                f"--number_envs {n_envs} "
                                f"--total_timesteps {number_steps} "
                                f"--change_size {'True' if bools['change_size'] else 'False'} "
                                f"--change_mass {'True' if bools['change_mass'] else 'False'} "
                                f"--change_shape {'True' if bools['change_shape'] else 'False'} "
                                f"--random {randomness}",
                                shell=True
                                )
                 # test(s, n_envs, number_steps, b["change_shape"], b["change_size"], b["change_mass"], randomness)

    print("\n\n\n\n\n\n\nAll done")