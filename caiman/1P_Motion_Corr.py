

import sys
import numpy as np
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import matplotlib.pyplot as plt
import cv2

import json

def read_params(filename):
    """
    Read the parameters of the motion correction from a json file, convert the json arrays to tuples, add the 'fnames' parameter and pass down the dictionary to a 'CNMFparams' object

    Parameters
    ----------
    filename : str
    The location of the json file containing the parameters

    Returns
    -------
    The function returns a 'CNMFparams' object containing the motion correction parameters
    """
    f = open(filename, 'r')
    mc_dict = json.load(f)
    f.close()
    for key in mc_dict:
        if isinstance(mc_dict[key], list):
            mc_dict[key] = totuple(mc_dict[key])
    mc_dict["fnames"] = fnames
    opts = params.CNMFParams(params_dict=mc_dict)

    return opts


def totuple(arr):
    """
    Try to turn the input into a tuple

    Parameters
    ----------
    arr : optional
    The object we try to turn into a tupple

    Returns
    -------
    The function either returns the original input or the input in tupple format depending on if we succeeded in converting the input into a tupple
    """
    try:
        return tuple(totuple(f) for f in arr)
    except TypeError:
        return arr

def set_up_cluster():
    """
    Start a cluster for parallel processing, if a cluster is already used it will be closed and a new session will be opened

    To enable parallel processing a (local) cluster needs to be set up. If dview = dview is used
    in the downstream analysis then parallel processing will be applied, if dview = None is used then
    no parallel processing will be employed.

    Returns
    -------
    dview
        expresses the cluster option

    n_processes
        number of workers in dview

    c
        iparallel.Client object, only used when backend is not 'local'
    """
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=None,
                                                     single_thread=False)

    return c, dview, n_processes


def motion_correction(save_movie):
    """
    Perform motion correction
    Parameters
    ----------
    save_movie
        Flag to save the correction matrix in a mmap file

    Returns
    -------
    mc
        The motion correction object
    """

    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct()

    return mc


def play_movie():
    #save template?
    #show movies

    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
    bord_px = 0 if border_nan is 'copy' else bord_px
    if save_movie_mmap:
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C', border_to_0=bord_px)
    if save_movie_tif: #search for the saving template
        save_name = mc.fname_tot_rig
        cm.load(mc.mmap_file).save()

def show_motion_corr_movie():



def show_plots():



def show_min_max_shifts():



def save_mmap():



if __name__ == "__main__":
    #original_stdout = sys.stdout
    #with open('C:/Users/Martin/Desktop/Motion_correction_for/Motion_correction_offsets.txt', 'w') as f:
        #sys.stdout = f
        my_lst = ['/media/hillierlab/T7/Data/Experiment/2021/08.03/data/20210803-110946-Fluorescent-recording.tif',
                  '/media/hillierlab/T7/Data/Experiment/2021/08.03/data/20210803-111139-Fluorescent-recording.tif',
                  '/media/hillierlab/T7/Data/Experiment/2021/08.03/data/20210803-111355-Fluorescent-recording.tif',
                  '/media/hillierlab/T7/Data/Experiment/2021/08.03/data/20210803-111641-Fluorescent-recording.tif',
                  '/media/hillierlab/T7/Data/Experiment/2021/08.03/data/20210803-111820-Fluorescent-recording.tif',
                  '/media/hillierlab/T7/Data/Experiment/2021/08.03/data/20210803-123810-Fluorescent-recording.tif',
                  '/media/hillierlab/T7/Data/Experiment/2021/08.03/data/20210803-124444-Fluorescent-recording.tif',
                  '/media/hillierlab/T7/Data/Experiment/2021/08.03/data/20210803-124546-Fluorescent-recording.tif'
                  ]

        for x in range(8):
            path = my_lst[x]
            print(x+1,". recording\n")
            fnames = [download_demo(path)]
            #set up clusters for parallel processing
            if 'dview' in locals():
                cm.stop_server(dview=dview)
            c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)

            # dataset dependent parameters
            frate = 35                       # movie frame rate
            decay_time = 0.4                 # length of a typical transient in seconds

            # motion correction parameters
            motion_correct = True    # flag for performing motion correction
            pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
            gSig_filt = (3, 3)       # size of high pass spatial filtering, used in 1p data
            max_shifts = (5, 5)      # maximum allowed rigid shift
            strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
            overlaps = (24, 24)      # overlap between pathes (size of patch strides+overlaps)
            max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
            border_nan = 'copy'      # replicate values along the boundaries

            mc_dict = {
                'fnames': fnames,
                'fr': frate,
                'decay_time': decay_time,
                'pw_rigid': pw_rigid,
                'max_shifts': max_shifts,
                'gSig_filt': gSig_filt,
                'strides': strides,
                'overlaps': overlaps,
                'max_deviation_rigid': max_deviation_rigid,
                'border_nan': border_nan
            }

            opts = params.CNMFParams(params_dict=mc_dict)

            if motion_correct:
                # do motion correction rigid
                mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
                mc.motion_correct(save_movie=True)
                fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig

                #x_axis_elements = [x_shift[0] for x_shift in mc.shifts_rig]
                #y_axis_elements = [y_shift[1] for y_shift in mc.shifts_rig]
                """
                print("Max x oriented shift(px): ", np.max(x_axis_elements))
                print("Max y oriented shift(px)? ", np.max(y_axis_elements))

                print("Min x oriented shift(px): ", np.min(x_axis_elements))
                print("Min y oriented shift(px)? ", np.min(y_axis_elements), "\n\n")
                """

                #tmp_filename_x = "x_direction_offset_for_" + str(x) + ".recordnig.png"
                #tmp_filename_y = "y_direction_offset_for_" + str(x) + ".recordnig.png"

                """
                x_fig = plt.figure()
                plt.plot(x_axis_elements)  # % plot rigid shifts
                plt.legend('x shifts')
                plt.xlabel('frames')
                plt.ylabel('pixels')
                #plt.show(block=True)
                plt.savefig(tmp_filename_x)

                y_fig = plt.figure()
                plt.plot(y_axis_elements)  # % plot rigid shifts
                plt.legend('y shifts')
                plt.xlabel('frames')
                plt.ylabel('pixels')
                # plt.show(block=True)
                plt.savefig(tmp_filename_y)
                """
                tmp_movie_name = str(x+1) + ".recording_motion_corrected.tif"
                cm.load(mc.mmap_file).save(tmp_movie_name)
        #sys.stdout = original_stdout



