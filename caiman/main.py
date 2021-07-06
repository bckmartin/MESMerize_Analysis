from MyCaimanModules import averaging, slice_n
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Arr = np.array([1, 3, 2, 6, 7, 4, 2, 5, 7, 2, 4, 6, 2, 5, 7, 2, 5, 6, 2, 3, 5, 67, 2, 4, 5, 6, 2, 3, 5, 6, 1, 6, 8, 2, 3, 7])
    n = 4
    L = slice_n(Arr, n)
    Avg_arr = averaging(L)
    print(Avg_arr[0])
    print(Avg_arr[1])
    print(Avg_arr[2])
    print(Avg_arr[3])
    plt.plot(Avg_arr)
    plt.show(block=True)
