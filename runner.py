import sys, os
import numpy as np
import uuid
import pickle
import multiprocessing as mp
from tqdm import tqdm
from yaml import safe_load
from copy import deepcopy
from simulator import Simulator
from scipy.stats import norm


def read_arguments():
    try:
        file_name: str = "inputs/" + str(sys.argv[1]) + ".yaml"
    except:
        raise ValueError(
            "No filename provided. Please run as 'python main.py <filename>'"
        )

    with open(file_name, "r") as f:
        data = safe_load(f)

    try:
        type = data["simulation"]["type"]
        game = data["simulation"]["game"]
        return data["running"], (data["simulation"], data[game]), data[type]
    except:
        raise ValueError(f"Variable '{type}' or '{game}' are not defined.")


def prepare_configurations(runs, sim_args):
    simulation, parameters = sim_args
    size = simulation["population_size"]
    path = f"./outputs/{uuid.uuid1()}"
    os.mkdir(path)
    simulation["output"] = path

    k_values = np.linspace(0, size, runs, dtype=int)

    configurations = []
    for i in range(runs):
        simulation["init_k"] = k_values[i]
        configurations.append((deepcopy(simulation), parameters))

    return path, configurations


def collapse_results(path, save_data):
    dist = []
    efcs = np.array([])
    
    for file in os.listdir(path):
        with open(f"{path}/{file}", 'rb') as handle:
            data = pickle.load(handle)
            dist.append(data[:-1])
            efcs = np.append(efcs, data[-1])

    dist = np.array(dist).T

    mus = np.array([])
    stds = np.array([])
    for point in dist:
        mu, std = norm.fit(point)
        mus = np.append(mus, mu)
        stds = np.append(stds, std)

    efc_mu, efc_std = norm.fit(efcs)

    if save_data:
        new_file = f"{path}.pickle" 
        with open(new_file, 'wb') as handle:
            pickle.dump([mus, stds, efc_mu, efc_std], handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved cleaned data of distribution to {new_file}.")

    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
    os.rmdir(path)
    
    return mus, stds, efc_mu, efc_std


def run_simulation(args):
    simulation, parameters = args
    sim = Simulator(simulation, parameters)

    sim.run()


def run_simulations(run_args, sim_args, save_data=True):
    runs, cores = run_args["runs"], run_args["cores"]

    path, configurations = prepare_configurations(runs, sim_args)

    num_cores = mp.cpu_count() - 1 if cores == "all" else cores

    print("=" * 10 + f" Running {runs} simulations in {num_cores} cores " + "=" * 10)

    print("Pooling processes...")
    with mp.Pool(processes=num_cores) as pool:
        list(
            tqdm(
                pool.imap(run_simulation, configurations),
                total=runs,
            )
        )

    print("Simulations done. Processing results...")

    return collapse_results(path, save_data)