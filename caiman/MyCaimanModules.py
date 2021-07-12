import imageio
from glob import glob
import numpy as np




def osszefuzes(img_loc):
    """Concatenate multiple tiff images into a movie

    :param img_loc: The locations of the images
    :return: The concatenated movie
    """
    files = sorted(glob(img_loc))
    imgs = np.array([imageio.imread(f) for f in files])

    return imgs


def averaging(signal_list):
    """Make an average plot out of a list of plots

    The plots need to be the same length

    :param signal_list: The list containing the plots that should be averaged
    :return: The averaged array
    """
    #check if their length match
    N = len(signal_list)-1
    len_0 = len(signal_list[0])
    i = 1
    while (i <= N):
        if (len_0 == len(signal_list[i])):
            i = i+1
        else:
            break
    same_length = (i > N)

    #loop of averaging frame by frame
    if same_length:
        avg_arr = np.zeros(signal_list[0].shape, dtype=float)
        for i in range(len(signal_list[0])):
            sum = 0
            for j in range(len(signal_list)):
                sum = sum + signal_list[j][i]
            avg_arr[i] = sum/len(signal_list)
    else:
        print("The signals should be uniform in length")
        avg_arr = None

    return avg_arr

def slice_n(Arr, n):
    """Check if the numpy array can be sliced up into n long arrays then slice them up and store
        store them in a list.

    :param Arr: The numpy array to be sliced up into n long pieces.
    :param n: The length of the result arrays.
    :return: A list of the sliced up arrays.
    """
    if (Arr.shape[0] % n) == 0:
        Lst = []
        stim_rep = int(Arr.shape[0] / n)
        for i in range(stim_rep):
            tmp_arr = Arr[i * n:i * n + n]
            Lst.append(tmp_arr)
    return Lst

def averaging_2(Arr, stim_len):
    """Check if the numpy array can be sliced up into stim_len long arrays then average them.

    :param Arr: The numpy array to be averaged on stim_len long stimuli repetitions.
    :param stim_len: The length of one stimulus.
    :return: The averaged array
    """
    l = Arr.shape[0]
    if (Arr.shape[0] % stim_len) == 0:
        stim_rep = l/stim_len
        Arr_sum = np.zeros(stim_len)
        for i in range(int(stim_len)):
            Arr_sum[i] = Arr[i::stim_len].sum()

    return Arr_sum/stim_rep
