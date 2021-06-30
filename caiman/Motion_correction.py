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
from caiman.utils.visualization import plot_contours, view_patches, plot_contours


def set_up_log():
    """Create a log for the warnings occurring during running

    :return:
    """
    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename="/tmp/caiman.log",
                        level=logging.WARNING)


def select_files(path):
    """Select one or multiple file(s) and download them

    :param path: The url(s) of the file(s) selected
    :return: The path of the downloaded file(s)
    """
    fnames = [download_demo(path)]
    return fnames


def play_movie(fnames):
    """ Play a movie

    :param fnames: The movie to be played
    :return:
    """
    m_orig = cm.load_movie_chain(fnames)
    ds_ratio = 0.2
    m_orig.resize(1, 1, ds_ratio).play(
        q_max=99.5, fr=30, magnification=2)


def basic_plots(fnames):
    """ Plot the mean and std of a movie

    :param fnames: The movie from which mean and std is being calculated
    :return:
    """
    movie = cm.load_movie_chain(fnames)
    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    plt.imshow(np.mean(movie, 0))
    plt.title('mean')
    fig.add_subplot(2, 2, 2)
    plt.imshow(np.std(movie, 0))
    plt.title('std')
    fig.add_subplot(2, 2, 3)
    plt.plot(np.mean(movie, axis=(1, 2)))
    plt.title('mean2')
    plt.show(block=True)


def correlation_image(fnames):
    """Show the correlation image of a movie

    :param fnames: The movie from which the correlation image is being calculated
    :return:
    """
    movie = cm.load_movie_chain(fnames)
    ci = movie.local_correlations(eight_neighbours=True, swap_dim=False)
    #t,x,y formatnal, ha a time dimenzio van a vegen akkor swap_dim = true
    plt.imshow(ci)
    plt.title('correlation image')
    plt.show(block=True)


def setup_params(
    fr = 30,
    decay_time = 0.4,

    strides = (48, 48),
    overlaps = (24, 24),
    max_shifts = (6, 6),
    max_deviation_rigid = 3,
    pw_rigid = True,
    p = 1,
    gnb = 2,
    merge_thr = 0.85,
    rf = 15,
    stride_cnmf = 6,
    K = 4,
    gSig = [4, 4],
    method_init = 'greedy_roi',
    ssub = 1,
    tsub = 1,
    min_SNR = 2.0,
    rval_thr = 0.85,
    cnn_thr = 0.99,
    cnn_lowest = 0.1,


        ):
    """Set the parameters for motion correction, source extraction, deconvolution and component evaluation

    :param fr: imaging rate in frames per second
    :param decay_time: length of a typical transient in seconds

    Motion correction parameters
    :param strides: start a new patch for pw-rigid motion correction every x pixels
    :param overlaps: overlap between pathes (size of patch strides+overlaps)
    :param max_shifts: maximum allowed rigid shifts (in pixels)
    :param max_deviation_rigid: maximum shifts deviation allowed for patch with respect to rigid shifts
    :param pw_rigid: flag for performing non-rigid motion correction

    Source extraction and deconvolution parameters
    :param p: order of the autoregressive system
    :param gnb: number of global background components
    :param merge_thr: merging threshold, max correlation allowed
    :param rf: half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    :param stride_cnmf: amount of overlap between the patches in pixels
    :param K: number of components per patch
    :param gSig: expected half size of neurons in pixels
    :param method_init: initialization method (if analyzing dendritic data using 'sparse_nmf')
    :param ssub: spatial subsampling during initialization
    :param tsub: temporal subsampling during initialization

    Component evaluation parameters
    :param min_SNR: signal to noise ratio for accepting a component
    :param rval_thr: space correlation threshold for accepting a component
    :param cnn_thr: threshold for CNN based classifier
    :param cnn_lowest: neurons with cnn probability lower than this value are rejected
    :return: Dictionary of the set parameters
    """
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
                 'gSig': gSig,
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
    """Srart a cluster for parallel processing, if a cluster is already used it will be closed and a new session will be opened

    To enable parallel processing a (local) cluster needs to be set up. If dview = dview is used
    in the downstream analysis then parallel processing will be applied, if dview = None is used then
    no parallel processing will be employed.

    :return: dview: expresses the cluster option
             n_processes: number of workers in dview
             c: iparallel.Client object, only used when backend is not 'local'
    """
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    return c, dview, n_processes


def NoRMCorre(mc):
    """Run piece-wise rigid motion correction using NoRMCorre

    :param mc: The motion correction object
    :return:m_els: The motion corrected movie
            border_to_0: The maximum shift to be used in trimming against NaNs
    """
    mc.motion_correct(save_movie=True)
    m_els = cm.load(mc.fname_tot_els)
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
    return m_els, border_to_0


def play_2_movies(fnames, m_els):
    """Play two movies side by side

    :param fnames: Movie1 to play
    :param m_els:  Movie2 to play
    :return:
    """
    m_orig = cm.load_movie_chain(fnames)
    ds_ratio = 0.2
    cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                    m_els.resize(1, 1, ds_ratio)],
                   axis=2).play(fr=60, gain=15, magnification=2, offset=0)  # press q to exit


def memory_map(mc, border_to_0, dview):
    """Memory map the files in order 'C'

    :param mc: The motion corrected object
    :param border_to_0:
    :param dview: expresses the cluster option
    :return: images: The memory mapped variable reshaped
             dims: the dimensions of the FOV
    """
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0, dview=dview)  # exclude borders
    Yr, dims, T = cm.load_memmap(fname_new)

    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    return images, dims


def restart_cluster(dview):
    """Close the currently running cluster and start a new one

    :param dview: expresses the cluster option
    :return: dview: expresses the cluster option
             n_processes: number of workers in dview
             c: iparallel.Client object, only used when backend is not 'local'
    """
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    return c, dview, n_processes


def CNMF(n_processes, opts, dview, images):
    """Run CNMF on patches

    The FOV is split into different overlapping patches that are subsequently processed in parallel by
    the CNMF algorithm. The results from all the patches are merged. The results are the refined by
    additional CNMF iterators.

    :param n_processes: number of workers used
    :param opts: parameters specified
    :param dview: expresses the cluster option
    :param images: The memory mapped object
    :return: An estimate object with the identified ROIs and their attributes
    """
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    return cnm


def see_results(images, cnm):
    """Visualize the result of the ROI extraction

    :param images: The memory mapped variable
    :param cnm: The result of the CNMF algorithm
    :return: An image of the contours of the identified ROIs
    """

    cnm.estimates.plot_contours()
    #cnm.estimates.view_components()
    plt.show(block=True)


def seeded_cnmf(cnm, images, dview):
    """Re-run seeded cnmf on accepted patches

    :param cnm: The result of the CNMF algorithm
    :param images: Memory mapped variables
    :param dview: The cluster option
    :return: the result of the seeded cnmf
    """
    cnm2 = cnm.refit(images, dview=dview)
    return cnm2


def comp_eval(cnm2, images, dview):
    """ Evaluate and filter the components

    The component evaluation is based on three factors:
    The shape of the component must correspond to the data of the given location.
    A minimum peak SNR is required over the length of transient.
    Each shape has to pass a CNN based classifier.

    :param cnm2: The result of the seeded CNMF algorithm
    :param images: Memory mapped variables
    :param dview: The cluster option
    :return: An estimate object with the filtered out components.
    """
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    return cnm2


def stop_cluster(dview):
    """ Stop the running cluster to clear memory

    :param dview: The cluster option
    :return:
    """
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)


if __name__ == "__main__":

    #set_up_log()


    path = 'C:/Users/Martin/Documents/Onlab/Fmr1_KO_P9-P11_4Hz/Movie_1.tif'
    #path = 'Sue_2x_3000_40_-46.tif'
    fnames = select_files(path)




    #print("\n Showing basic plots...")
    #basic_plots(fnames)
    #print("\n Showing correlation image...")
    #correlation_image(fnames)

    #print("\n playing the original movie...")
    #play_movie(fnames)
    opts = setup_params()      #parameterek
    c, dview, n_processes = setup_cluster()
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))     #create a motion correction object with the specified parameters
    m_els, border_to_0 = NoRMCorre(mc)
    #print("\n Playing the original and the motion corrected movie...")
    #play_2_movies(fnames, m_els)
    images, dims = memory_map(mc, border_to_0, dview)


    c, dview, n_processes = restart_cluster(dview)
    cnm = CNMF(n_processes, opts, dview, images)

    cnm.estimates.detrend_df_f()
    print(type(cnm.estimates.F_dff[0]))

    """
    #print("\n Showing contours of found ROIs...")
    #see_results(images,cnm)
    cnm2 = seeded_cnmf(cnm, images, dview)
    cnm2 = comp_eval(cnm2, images, dview)
    stop_cluster(dview)
    #print("\n Showing contours of found ROIs after evaluation...")
    #see_results(images,cnm2)
"""
    print("done")



