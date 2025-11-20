from tkinter import Label
from matplotlib import pyplot as plt
import numpy as np

def plot_graph(arr1, name, arr2=None, l1=None, l2=None):
    plt.plot(range(1, 21), arr1, label=l1, marker='o')
    if arr2 is not None:
        plt.plot(range(1,21), arr2, label=l2, marker='o')
    plt.title(name + ' vs. k')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 21, 1))
    plt.ylabel(name)
    plt.savefig(f"Plots/{name.replace(' ', '_')}_vs_k")
    plt.legend()
    plt.show()
    


