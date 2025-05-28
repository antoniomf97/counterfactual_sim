import os
import sys
import uuid
import pickle
import multiprocessing as mp

from tqdm import tqdm
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
        return data["running"], (data["simulation"], data[data["simulation"]["game"]])
    except:
        raise ValueError(f"Game '{data['population']['game']}' not defined.")


def run_simulation(args):
    simulation, parameters = args
    sim = Simulator(simulation, parameters)
    
    sim.run()


def run_simulations(run_args, sim_args, clear_data=True):
    outdir = f"./outputs/{uuid.uuid1()}"
    os.mkdir(outdir)
    sim_args[0]["outdir"] = outdir
    
    num_cores = mp.cpu_count() - 1 if run_args["cores"] == "all" else run_args["cores"]

    print("============ Running experiment of", run_args["runs"],
          "simulations in", num_cores, "cores: ============")

    print("Pooling processes...")
    with mp.Pool(processes=num_cores) as pool:
        list(
            tqdm(pool.imap(run_simulation, [sim_args] * run_args["runs"]), total=run_args["runs"])
        )

    print("Simulations done. Processing results...")


if __name__ == "__main__":
    run_args, sim_args = read_arguments()
    run_simulation(sim_args)
    