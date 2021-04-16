
import imageio
from glob import glob
import numpy as np

def osszefuzes(img_loc):
    files = sorted(glob(img_loc))
    imgs = np.array([imageio.imread(f) for f in files])

    return imgs


if __name__ == "__main__":
    img_loc = "/home/ulbertlab/Documents/Beck_Martin_Bsc/neurofinder_00_01_dataset_mesmerize/neurofinder.00.01/images/*.tiff"
    #kep = osszefuzes(img_loc)
    path = "/home/ulbertlab/Documents/Beck_Martin_Bsc/neurofinder_00_01_dataset_mesmerize/neurofinder.00.01/kep.npy"
    #np.save(path, kep)
    kep = np.load(path)


