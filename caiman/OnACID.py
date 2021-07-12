import logging
import numpy as np

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
import matplotlib.pyplot as plt


def set_up_log_online():
    """Set up a log for the program

    :return: -
    """
    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename="/tmp/caiman.log",
                        level=logging.INFO)


def download_files(fname, fld_name):
    """Download the files to be processed

    :param fname: The files we want to process
    :param fld_name: The folder we want to store them in
    :return: A list of the stored filepaths
    """
    fnames = [download_demo(f, fld_name) for f in fname]
    """This is the solution in the demo
    fnames = []
    fnames.append(download_demo(fname[0], fld_name))
    fnames.append(download_demo(fname[1], fld_name))
    fnames.append(download_demo(fname[2], fld_name))
    """
    return fnames


def setup_params_online(
    fr = 15,
    decay_time = 0.5,
    gSig = (4,4),
    p = 1,
    min_SNR = 1,
    rval_thr = 0.90,
    ds_factor = 1,
    gnb = 2,
    mot_corr = True,
    pw_rigid = False,
    sniper_mode = True,
    init_batch = 200,
    expected_comps = 500,
    dist_shape_update = True,
    min_num_trial = 10,
    K = 2,
    epochs = 2,
    show_movie = False ):
    """Set up the values for the parameters

    :param fr: frame rate (Hz)
    :param decay_time: approximate length of transient event in seconds
    :param gSig: expected half size of neurons
    :param p: order of AR indicator dynamics
    :param min_SNR: minimum SNR for accepting new components
    :param rval_thr: correlation threshold for new component inclusion
    :param ds_factor: spatial downsampling factor (increases speed but may lose some fine structure)
    :param gnb: number of background components
    :param mot_corr: flag for online motion correction
    :param pw_rigid: flag for pw-rigid motion correction (slower but potentially more accurate)
    :param sniper_mode: flag using a CNN to detect new neurons (o/w space correlation is used)
    :param init_batch: number of frames for initialization (presumably from the first file)
    :param expected_comps: maximum number of expected components used for memory pre-allocation (exaggerate here)
    :param dist_shape_update: flag for updating shapes in a distributed way
    :param min_num_trial: number of candidate components per frame
    :param K: initial number of components
    :param epochs: number of passes over the data
    :param show_movie: flag for showing the movie
    :return: opts: The parameter object to be used in later computations
    """

    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int'))     # recompute gSig if downsampling is involved
    max_shifts_online = np.ceil(10. / ds_factor).astype('int')          # maximum allowed shift during motion correction
    params_dict = {'fnames': fnames,
               'fr': fr,
               'decay_time': decay_time,
               'gSig': gSig,
               'p': p,
               'min_SNR': min_SNR,
               'rval_thr': rval_thr,
               'ds_factor': ds_factor,
               'nb': gnb,
               'motion_correct': mot_corr,
               'init_batch': init_batch,
               'init_method': 'bare',           #bare, cnmf, seeded
               'normalize': True,
               'expected_comps': expected_comps,
               'sniper_mode': sniper_mode,
               'dist_shape_update' : dist_shape_update,
               'min_num_trial': min_num_trial,
               'K': K,
               'epochs': epochs,
               'max_shifts_online': max_shifts_online,
               'pw_rigid': pw_rigid,
               'show_movie': show_movie}
    opts = cnmf.params.CNMFParams(params_dict=params_dict)

    return opts


def CNMF_online(opts):
    """Make an OnACID object and run the entire online pipeline

    :param opts: Parameters for the algorithm
    :return: cnm: the result object of the algorithm
    """
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()

    return cnm


def see_results_online(cnm):
    """Show the result of the CNMF online algorithm

    :param cnm: The result object of the CNMF online algorithm
    :return: -
    """
    logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    Cn = cm.load(fnames[0], subindices=slice(0, 500)).local_correlations(swap_dim=False)
    cnm.estimates.plot_contours()
    plt.show(block=True)

def plot_run_time(cnm):
    """Show the motion, detect, online and shapes

    :param cnm: The result object of the CNMF online algorithm
    :return: -
    """
    T_motion = 1e3 * np.array(cnm.t_motion)
    T_detect = 1e3 * np.array(cnm.t_detect)
    T_shapes = 1e3 * np.array(cnm.t_shapes)
    T_online = 1e3 * np.array(cnm.t_online) - T_motion - T_detect - T_shapes
    plt.figure()
    plt.stackplot(np.arange(len(T_motion)), T_motion, T_online, T_detect, T_shapes)
    plt.legend(labels=['motion', 'process', 'detect', 'shapes'], loc=2)
    plt.title('Processing time allocation')
    plt.xlabel('Frame #')
    plt.ylabel('Processing time [ms]')
    plt.ylim([0, 140])
    plt.show(block=True)


if __name__ == "__main__":

    #set_up_log_online()
    fld_name = 'Mesoscope'
    fname = ['Tolias_mesoscope_1.hdf5', 'Tolias_mesoscope_2.hdf5', 'Tolias_mesoscope_3.hdf5']
    fnames = download_files(fname, fld_name)
    opts = setup_params_online()
    cnm = CNMF_online(opts)
    #print("The results of the CNMF online algorithm: ")
    #see_results_online(cnm)
    #print("The runtime of the algorithm: ")
    #plot_run_time(cnm)


    print("Done")