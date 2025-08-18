import numpy as np
import pickle
from copy import deepcopy
from runner import read_arguments, run_simulations
from plotting import plot_stat_dist_fill_between


# def e1_depth(maxdepth = 5, enhancement = (5,7), step = None):
#     run_args, sim_args = read_arguments()
#     depths = [[1.]]
#     for n in range(1, maxdepth + 1):
#         depth = np.zeros(n + 1)
#         depth[0] = 0.9
#         depth[-1] = 0.1
#         depths.append(depth)
    
#     s = step if step else maxdepth + 1
#     enhancements = np.linspace(enhancement[0], enhancement[1], s)

#     sim_args_all = []

#     for f in enhancements:
#         aux = []
#         for depth in depths:
#             sim_args[0]["depth_distribution"] = depth
#             sim_args[1]["F"] = f
#             aux.append(deepcopy(sim_args))
#         sim_args_all.append(aux)
 
#     efcs_avg = []
#     efcs_all = []
#     for i, f in enumerate(enhancements):
#         aux_avg = []
#         aux_all = []
#         for j, depth in enumerate(depths):
#             print("=" * 8 + f" Simulations with enhancement={f} depth={j} ")
#             efc, efc_all = run_simulations(run_args, sim_args_all[i][j])
#             aux_avg.append(efc)
#             aux_all.append(efc_all)
#         efcs_avg.append(aux_avg)
#         efcs_all.append(aux_all)

#     r = run_args["runs"]
#     g = sim_args[0]["generations"]
#     file = f"./outputs/e1_depth/e1_depth_{len(enhancements)}_{len(depths)}_{r}_{g}.pickle"
#     with open(file, 'wb') as handle:
#         pickle.dump(efcs_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         print(f"Saved cleaned data to {file}")

#     return efcs_all, enhancements


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



def e1_stationary_distribution():
    run_args, sim_args, sd_args = read_arguments()

    depths_list = sd_args["depths"]
    prob_imitation = sd_args["prob_imitation"]

    depths = [[1.]]
    for d in depths_list:
        depth = np.zeros(d + 1)
        depth[0] = prob_imitation
        depth[-1] = 1 - prob_imitation
        depths.append(depth)

    sim_args_all = []
    for depth in depths:
        sim_args[0]["depth_distribution"] = depth
        sim_args_all.append(deepcopy(sim_args))  

    all_mus = []
    all_stds = []
    for j, depth in enumerate(depths):
        print("=" * 8 + f" Simulations with depth={j} ")
        mus, stds, efc_m, _ = run_simulations(run_args, sim_args_all[j], save_data=sd_args["save_data"])
        all_mus.append(mus)
        all_stds.append(stds)

    print("efc", efc_m)

    path = "./outputs/e1_sd/" + get_path(run_args, sim_args, sd_args)

    plot_stat_dist_fill_between(all_mus, all_stds, labels=[0]+depths_list, save_fig=sd_args["save_fig"], path=path)

if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()
    e1_stationary_distribution()
    end = perf_counter()
    print(f"Total time elapsed = {end - start} s")
