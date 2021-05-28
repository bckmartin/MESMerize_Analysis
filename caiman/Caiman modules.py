import caiman as cm
import numpy as np
import matplotlib.pyplot as plt
#load movie
def load_movie(path):
    single_movie = cm.load(path)
    print(single_movie.shape)
    return single_movie

#lehetne tobb filmet is egymas utan betolteni

#play movie
def play_movie(movie):
    movie.play(magnification=2, fr=30, q_min=0.1, q_max=99.75)
#q-val megall

#numpy informaciok matplotlib plotokon
def basic_plots(movie):
    plt.imshow(np.mean(movie, 0))
    plt.imshow(np.std(movie, 0))
    plt.plot(np.mean(movie, axis=(1, 2)))

#correlacios plot
def correlation_image(movie):
    CI = movie.local_correlations(eight_neighbours=True, swap_dim=False)    #t,x,y formatnal, ha a time dimenzio van a vegen akkor swap_dim = true
    pl.imshow(CI)


                                    ### Motion correction ###

from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div

import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)

import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo

#a filet nem muszaj betolteni a RAM-ba ahhoz, hogy motion correctiont vegezzunk rajtuk

# If you adapt this demo for your data make sure to pass the complete path to your file(s). Remember to pass the fname variable as a list.

fnames = ['Movie_1.tif']        #relative path, lehet kesobb absolute kell
m_orig = cm.load_movie_chain(fnames)
downsample_ratio = .2  # motion can be perceived better when downsampling in time
#m_orig.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=30, magnification=2) lejatszas

#parameters
max_shifts = (6, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
num_frames_split = 100  # length in frames of each chunk of the movie (to be processed in parallel)
max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

#commands needed to be executed from the terminal

#export MKL_NUM_THREADS = 1
#export OPENBLAS_NUM_THREADS=1

#clustering

if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# create a motion correction object
mc = MotionCorrect(fnames, dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid,
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan)

#rigid motion correction és utána mentés
mc.motion_correct(save_movie=True)

#elasztik motion correction az elozo memorymapja alapjan

mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)

mc.motion_correct(save_movie=True, template=mc.total_template_rig)
m_els = cm.load(mc.fname_tot_els)





