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
    bools = [# {"change_shape": True, "change_mass": False, "change_size": False},
             {"change_shape": False, "change_mass": True, "change_size": False},
             # {"change_shape": False, "change_mass": False, "change_size": True}
             ]
    multi_process = False
    max_processes_at_once = 1
    randomness = 0.005
    os.makedirs("results", exist_ok=True)

    if not multi_process:
        for b in bools:
             for s in range(seeds):
                 subprocess.run(
                                f"..\\venv\\Scripts\\python.exe ./test_td3.py "
                                f"--seed {s} "
                                f"--number_envs {n_envs} "
                                f"--total_timesteps {number_steps} "
                                f"--change_size {'True' if b['change_size'] else 'False'} "
                                f"--change_mass {'True' if b['change_mass'] else 'False'} "
                                f"--change_shape {'True' if b['change_shape'] else 'False'} ",
                                shell=True
                                )
                 # test(s, n_envs, number_steps, b["change_shape"], b["change_size"], b["change_mass"], randomness)


    print("\n\n\n\n\n\n\nAll done")