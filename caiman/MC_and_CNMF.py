import bokeh.plotting as bpl
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour

def unknown_module():
    try:
        cv2.setNumThreads(0)
    except():
        pass

    try:
        if __IPYTHON__:
            # this is used for debugging purposes only. allows to reload classes
            # when changed
            get_ipython().magic('load_ext autoreload')
            get_ipython().magic('autoreload 2')
    except NameError:
        pass

def set_up_log():
    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename="/tmp/caiman.log",
                        level=logging.WARNING)

def select_files(path):
    #fnames = path  # filename to be processed
    fnames = [download_demo(path)]
    return fnames

def play_movie(fnames):
    m_orig = cm.load_movie_chain(fnames)
    ds_ratio = 0.2
    m_orig.resize(1, 1, ds_ratio).play(
        q_max=99.5, fr=30, magnification=2)

def setup_params(
    fr = 30,  # imaging rate in frames per second
    decay_time = 0.4,  # length of a typical transient in seconds

    # motion correction parameters
    strides = (48, 48),  # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24),  # overlap between pathes (size of patch strides+overlaps)
    max_shifts = (6, 6),  # maximum allowed rigid shifts (in pixels)
    max_deviation_rigid = 3,  # maximum shifts deviation allowed for patch with respect to rigid shifts
    pw_rigid = True,  # flag for performing non-rigid motion correction

    # parameters for source extraction and deconvolution
    p = 1,  # order of the autoregressive system
    gnb = 2,  # number of global background components
    merge_thr = 0.85,  # merging threshold, max correlation allowed
    rf = 15,  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6,  # amount of overlap between the patches in pixels
    K = 4,  # number of components per patch
    gSig = [4, 4],  # expected half size of neurons in pixels
    method_init = 'greedy_roi',  # initialization method (if analyzing dendritic data using 'sparse_nmf')
    ssub = 1,  # spatial subsampling during initialization
    tsub = 1,  # temporal subsampling during intialization

    # parameters for component evaluation
    min_SNR = 2.0,  # signal to noise ratio for accepting a component
    rval_thr = 0.85,  # space correlation threshold for accepting a component
    cnn_thr = 0.99,  # threshold for CNN based classifier
    cnn_lowest = 0.1,  # neurons with cnn probability lower than this value are rejected


        ):
    opts_dict = {'fnames': fnames,
                 'fr': fr,
                 'decay_time': decay_time,
                 'strides': strides,
                 'overlaps': overlaps,
                 'max_shifts': max_shifts,
                 'max_deviation_rigid': max_deviation_rigid,
                 'pw_rigid': pw_rigid,
                 'p': p,
                 'nb': gnb,
                 'rf': rf,
                 'K': K,
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub,
                 'merge_thr': merge_thr,
                 'min_SNR': min_SNR,
                 'rval_thr': rval_thr,
                 'use_cnn': True,
                 'min_cnn_thr': cnn_thr,
                 'cnn_lowest': cnn_lowest}

    opts = params.CNMFParams(params_dict=opts_dict)

    return opts

def setup_cluster():
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    return c, dview, n_processes

def NoRMCorre(mc):
    mc.motion_correct(save_movie=True)
    m_els = cm.load(mc.fname_tot_els)
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
    return m_els, border_to_0

def play_2_movies(fnames, m_els):
    m_orig = cm.load_movie_chain(fnames)
    ds_ratio = 0.2
    cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                    m_els.resize(1, 1, ds_ratio)],
                   axis=2).play(fr=60, gain=15, magnification=2, offset=0)  # press q to exit
def memory_map(mc, border_to_0, dview):
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0, dview=dview)  # exclude borders
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    return images

def restart_cluster(dview):
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    return c, dview, n_processes

def CNMF(n_processes, opts, dview, images):
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    return cnm

def see_results():          #nb rossz helyen van
    Cn = cm.local_correlations(images.transpose(1, 2, 0))
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.plot_contours_nb(img=Cn)

"""
    lehet megegyszer futtatni
"""
if __name__ == "__main__":

    #unknown_module()
    #set_up_log()
    path = 'C:/Users/Martin/Documents/Onlab/Fmr1_KO_P9-P11_4Hz/Movie_1.tif'
    #path = 'Sue_2x_3000_40_-46.tif'
    fnames = select_files(path)
    play_movie(fnames)
    opts = setup_params()      #parameterek
    c, dview, n_processes = setup_cluster()
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    m_els, border_to_0 = NoRMCorre(mc)
    play_2_movies(fnames, m_els)
    images = memory_map(mc, border_to_0, dview)
    c, dview, n_processes = restart_cluster(dview)
    cnm = CNMF(n_processes, opts, dview, images)
    #see_results()
    
    print("done")



