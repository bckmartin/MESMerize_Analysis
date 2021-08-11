# demo file for applying the NoRMCorre motion correction algorithm on
# 1 - photon widefield imaging data
# Example file is provided from the miniscope project page
# www.miniscope.org
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

if __name__ == "__main__":
    original_stdout = sys.stdout
    with open('C:/Users/Martin/Desktop/Motion_correction_for/Motion_correction_offsets.txt', 'w') as f:
        sys.stdout = f
        my_lst = ['E:/Data/Experiment/2021/08.03/data/20210803-111355-Fluorescent-recording.tif',
                  'E:/Data/Experiment/2021/08.03/data/20210803-111641-Fluorescent-recording.tif',
                  'E:/Data/Experiment/2021/08.03/data/20210803-111820-Fluorescent-recording.tif',
                  'E:/Data/Experiment/2021/08.03/data/20210803-123810-Fluorescent-recording.tif',
                  'E:/Data/Experiment/2021/08.03/data/20210803-124444-Fluorescent-recording.tif',
                  'E:/Data/Experiment/2021/08.03/data/20210803-124546-Fluorescent-recording.tif']
        name_lst = ['C:/Users/Martin/Desktop/Motion_correction_for/3recording.png',
                    'C:/Users/Martin/Desktop/Motion_correction_for/4recording.png',
                    'C:/Users/Martin/Desktop/Motion_correction_for/5recording.png',
                    'C:/Users/Martin/Desktop/Motion_correction_for/6recording.png',
                    'C:/Users/Martin/Desktop/Motion_correction_for/7recording.png',
                    'C:/Users/Martin/Desktop/Motion_correction_for/8recording.png']
        for x in range(len(my_lst)):
            path = my_lst[x]
            print(x+3,". recording\n")
            #download data and convert to single precision
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
                mc.motion_correct()
                fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig

                x_axis_elements = [x_shift[0] for x_shift in mc.shifts_rig]
                y_axis_elements = [y_shift[1] for y_shift in mc.shifts_rig]

                print("Max x oriented shift(px): ", np.max(x_axis_elements))
                print("Max y oriented shift(px)? ", np.max(y_axis_elements))

                print("Min x oriented shift(px): ", np.min(x_axis_elements))
                print("Min y oriented shift(px)? ", np.min(y_axis_elements), "\n\n")



                fig = plt.figure()
                plt.plot(mc.shifts_rig)  # % plot rigid shifts
                plt.legend(['x shifts', 'y shifts'])
                plt.xlabel('frames')
                plt.ylabel('pixels')
                #plt.show(block=True)
                plt.savefig(name_lst[x])
        sys.stdout = original_stdout



