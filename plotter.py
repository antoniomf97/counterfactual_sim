import pickle
import matplotlib.pyplot as plt

filenames_82 = [
    "nshn100b10n10m5f8c10c00d1",
    "nshn100b10n10m5f8c10c00d82",
    "nshn100b10n10m5f8c10c00d802",
    "nshn100b10n10m5f8c10c00d8002",
    "nshn100b10n10m5f8c10c00d80002",
    "nshn100b10n10m5f8c10c00d80000000002",
    "nshn100b10n10m5f8c10c00d800000000000000000002",
]

filenames_cost_91 = [
    "nshn50b10n6m3f5c10c00d10",
    "nshn50b10n6m3f5c10c00d901",
    "nshn50b10n6m3f5c10c01d901",
    "nshn50b10n6m3f5c10c03d901",
    "nshn50b10n6m3f5c10c05d901",
]

filenames_5091 = [
    "nshn50b10n6m3f5c10c00d10",
    "nshn50b10n6m3f5c10c00d91",
    "nshn50b10n6m3f5c10c00d901",
    "nshn50b10n6m3f5c10c00d9001",
    "nshn50b10n6m3f5c10c00d900001",
    "nshn50b10n6m3f5c10c00d90000000001",
    "nshn50b10n6m3f5c10c00d900000000000000000001",
]


def run():
    data = {}
    idx = 0
    for filename in filenames_cost_91:
        with open("outputs/" + filename + ".pickle", "rb") as handle:
            data[idx] = pickle.load(handle)
            idx += 1

    labels = ["Immitation", "cost: 0.0", "cost: 0.1", "cost: 0.3", "cost: 0.5"]
    # labels = ["Immitation", "depth: 1", "depth: 2", "depth: 3", "depth: 5", "depth: 10", "depth: 20"]

    for i, values in data.items():
        plt.plot(values, label=labels[i])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
