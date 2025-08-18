import numpy as np
import matplotlib.pyplot as plt


colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def plot_stationary_distribution(Z, dist, path, save_fig, plot_show):
    plt.plot(dist)
    plt.xlim(0, Z)
    if save_fig:
        print("Figure saved.")
        plt.savefig(f"{path}.png")
    if plot_show:
        plt.show()


def plot_stat_dist_fill_between(mus_data, stds_data, labels, save_fig, path):
    x = np.array([i for i in range(0, len(mus_data[0]))])

    fig = plt.figure()
    for i in range(len(mus_data)):
        plt.plot(mus_data[i], "-", color=colors[i], label=f"Depth={labels[i]}")
        plt.fill_between(x, mus_data[i]-stds_data[i], mus_data[i]+stds_data[i], alpha=0.05, color=colors[i])
        plt.fill_between(x, mus_data[i]-2*stds_data[i], mus_data[i]+2*stds_data[i], alpha=0.05, color=colors[i])
    plt.ylim(0, 0.2)
    plt.grid()
    plt.ylabel("Stationary Distribution")
    plt.xlabel("Number of Cooperators")
    plt.legend()

    if save_fig:
        plt.savefig(f"{path}.png")
    else:
        plt.show()


def plot_heatmap(data, enhancements):
    muss = []
    for i in range(len(data)):
        mus = []
        for j in range(len(data[0])):
            mus.append(np.mean(data[i][j]))
        muss.append(mus)
    
    plt.imshow(muss)
    plt.colorbar()
    plt.yticks(range(len(data)), enhancements[::])
    plt.xlabel("Depth")
    plt.ylabel("Enhancement Factor")
    plt.show()