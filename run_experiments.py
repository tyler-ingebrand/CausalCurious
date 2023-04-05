import os
from multiprocessing import Pool, freeze_support
import time
from test_td3 import test

if __name__ == "__main__":
    freeze_support()

    n_envs = 32
    seeds = 5
    number_steps= 300_000
    bools = [{"change_shape": True, "change_mass": False, "change_size": False},
             {"change_shape": False, "change_mass": True, "change_size": False},
             {"change_shape": False, "change_mass": False, "change_size": True}
             ]
    multi_process = True
    max_processes_at_once = 3
    os.makedirs("results", exist_ok=True)

    if not multi_process:
        for b in bools:
             for s in range(seeds):
                 test(s, n_envs, number_steps, b["change_shape"], b["change_size"], b["change_mass"])

    else:
        args = []
        for b in bools:
            for s in range(seeds):
                this_arg = (s, n_envs, number_steps, b["change_shape"], b["change_size"], b["change_mass"])
                args.append(this_arg)

        while len(args) > 0:
            sub_args = [args.pop() for i in range(min(max_processes_at_once, len(args)))]
            with Pool(max_processes_at_once) as p:
                p.starmap(test, sub_args)

    print("\n\n\n\n\n\n\nAll done")