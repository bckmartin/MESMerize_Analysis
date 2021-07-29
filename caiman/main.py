import numpy as np
import matplotlib.pyplot as plt
from urllib import request
from MyCaimanModules import slice_n, averaging, averaging_2, averaging_by_frames

if __name__ == "__main__":
    #recieve streaming data
    #a = request.urlopen('https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/Tolias_mesoscope_1.hdf5')
    #print(a.__dict__)
    #print(a.headers)

    Arr = np.array(
        [1, 3, 2, 6,
         7, 4, 2, 5,
         7, 2, 4, 6,
         2, 5, 7, 2,
         5, 6, 2, 3,
         5, 67, 2, 4,
         5, 6, 2, 3,
         5, 6, 1, 6,
         8, 2, 3, 7])
    #n = 4
    start_frames = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32])
    x = averaging_by_frames(Arr, start_frames)
    print(x)
    """
    print(Avg_arr[0])
    print(Avg_arr[1])
    print(Avg_arr[2])
    print(Avg_arr[3])
    plt.plot(Avg_arr)
    plt.show(block=True)
    """




