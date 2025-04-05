import sys
import pickle

import numpy as np
from yaml import safe_load

from simulator import Simulator


def read_arguments():
    try:
        file_name: str = "inputs/" + str(sys.argv[1]) + ".yaml"
    except:
        raise ValueError(
            "No filename provided. Please run as 'python main.py <filename>'"
        )

    # Open and parse the YAML file
    with open(file_name, "r") as f:
        data = safe_load(f)

    try:
        return data["simulation"], {
            k: data[k] for k in ("population", "context", data["population"]["game"])
        }
    except:
        raise ValueError(f"Game '{data['population']['game']}' not defined.")


def run():
    sim_args, args = read_arguments()

    distribution = np.zeros(args["population"]["pop_size"] + 1)
    gens, runs, save = sim_args.values()

    for r in range(1, runs + 1):
        print(
            "Running simulation: " + "|" + r * "█" + (runs - r) * " " + f"|{r}/{runs}|"
        )
        s = Simulator(gens, **args)
        data, filename = s.run()
        distribution += data

    distribution = distribution / runs

    if save:
        with open("outputs/" + filename + ".pickle", "wb") as handle:
            pickle.dump(distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run()
