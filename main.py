import numpy as np
import pickle
import os
from copy import deepcopy
from runner import read_arguments, run_simulations
from plotting import plot_stat_dist_fill_between



# def e2_time_series(depths_list=()):
#     run_args, sim_args = read_arguments()

#     depths = [[1.]]
#     for d in depths_list:
#         depth = np.zeros(d + 1)
#         depth[0] = 0.9
#         depth[-1] = 0.1
#         depths.append(depth)

#     sim_args_all = []
#     for depth in depths:
#         sim_args[0]["depth_distribution"] = depth
#         sim_args_all.append(deepcopy(sim_args))

#     efcs_avg = []
#     efcs_all = []
#     for j, depth in enumerate(depths):
#         print("=" * 8 + f" Simulations with depth={j} ")
#         efc, efc_all = run_simulations(run_args, sim_args_all[j])
#         efcs_avg.append(efc)
#         efcs_all.append(efc_all)

#     r = run_args["runs"]
#     g = sim_args[0]["generations"]
#     file = f"./outputs/e2/e2_ts_{len(depths)}_{r}_{g}.pickle"
#     with open(file, 'wb') as handle:
#         pickle.dump(efcs_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         print(f"Saved cleaned data to {file}")

#     return efcs_all

def get_path(run_args, sim_args, other_args):
    r = run_args["runs"]
    t = "sd" if sim_args[0]["type"] == "stat_dist" else "hm"
    g = sim_args[0]["generations"]
    s = sim_args[0]["population_size"]
    gm = sim_args[0]["game"]

    if sim_args[0]["type"] == "stat_dist":
        d = "_".join(map(str, other_args["depths"]))
    else:
        pass

    par = "_".join(map(str, sim_args[1].values()))

    return f"{t}_{s}{gm}{par}_r{r}_g{g}_d{d}"



def e1_stationary_distribution(data_computed=False):
    run_args, sim_args, sd_args = read_arguments()

    dir = get_path(run_args, sim_args, sd_args)
    path = "./outputs/e1_sd/" + dir

    depths_list = sd_args["depths"]
    prob_imitation = sd_args["prob_imitation"]

    depths = [[1.]]
    for d in depths_list:
        depth = np.zeros(d + 1)
        depth[0] = prob_imitation
        depth[-1] = 1 - prob_imitation
        depths.append(depth)
    depths_list = [0] + depths_list

    all_mus = []
    all_stds = []
    all_efcs = []
    if data_computed:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} hasn't been created yet. First compute data.")
        
        order = []
        for file in os.listdir(path):
            if file.endswith(".png"):
                continue

            with open(f"{path}/{file}", 'rb') as handle:
                data = pickle.load(handle)
                order.append(data[0])
                all_mus.append(data[1])
                all_stds.append(data[2])
                all_efcs.append(data[3])

        if not order:
            raise ValueError(f"Files haven't been saved. First compute data.")

        order = sorted(range(len(order)), key=lambda i: order[i])
        all_mus = [all_mus[i] for i in order]
        all_stds = [all_stds[i] for i in order]
        all_efcs = [all_efcs[i] for i in order]
    else:
        sim_args_all = []
        for depth in depths:
            sim_args[0]["depth_distribution"] = depth
            sim_args_all.append(deepcopy(sim_args))  

        for j, depth in enumerate(depths):
            print("=" * 8 + f" Simulations with depth = {depths_list[j]} ")
            mus, stds, efcs, _ = run_simulations(run_args, sim_args_all[j], save_data=sd_args["save_data"], output_path=path)
            all_mus.append(mus)
            all_stds.append(stds)
            all_efcs.append(efcs)

    for i, efc in enumerate(all_efcs):
        print(f"Depth {depths_list[i]}: {efc}")

    plot_stat_dist_fill_between(all_mus, all_stds, labels=depths_list, save_fig=sd_args["save_fig"], path=path)


if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()
    e1_stationary_distribution(data_computed=False)
    end = perf_counter()
    print(f"Total time elapsed = {end - start} s")
