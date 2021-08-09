# demo file for applying the NoRMCorre motion correction algorithm on
# 1 - photon widefield imaging data
# Example file is provided from the miniscope project page
# www.miniscope.org

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

if __name__ == "__main__":

    #path = 'C:/Users/Martin/Downloads/20210729-141014-Fluorescent-recording.tif'

    #download data and convert to single precision
    #fnames = [download_demo(path)]
    fnames = [download_demo('msCam13.avi')]
    #set up clusters for parallel processing
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    # dataset dependent parameters
    frate = 30                       # movie frame rate
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
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
            plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
            plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
            plt.legend(['x shifts', 'y shifts'])
            plt.xlabel('frames')
            plt.ylabel('pixels')
            plt.show(block=True)
            print("rigid shift list index 1 type: ", type(mc.shifts_rig[0][0]))
            print("from the 1st item of the rigid shift list, the max of the 1st part of the tuple", np.max(mc.shifts_rig[0][0]))
            print("from the 1st item of the rigid shift list, the max of the 2nd part of the tuple", np.max(mc.shifts_rig[0][1]))

        bord_px = 0 if border_nan is 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                   border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(fnames, base_name='memmap_',
                                   order='C', border_to_0=0, dview=dview)

    inspect_results = True
    if inspect_results:
        cm.concatenate((cm.load(fnames), cm.load(mc.mmap_file)), axis=0).play()

    #cm.load(mc.mmap_file)