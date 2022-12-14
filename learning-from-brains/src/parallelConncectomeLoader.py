#!/usr/bin/env python3

import os
from typing import List, Union
import tarfile
import numpy as np
import pandas as pd
import argparse
from shutil import copyfile
from nilearn import plotting
import nilearn as nl
from nilearn.datasets import MNI152_FILE_PATH
import matplotlib.pyplot as plt
#from src.graph.connectome import Connectome
from src.graph.convConnectome import Connectome
from dirname import basePath

#from src.batcher.base import BaseBatcher, _pad_seq_right_to_n
import nibabel as nb
import mne
from typing import Dict
import torch
import glob

def loadConnectome(sub,task,run,sample) -> Dict[str, torch.tensor]:
    #def loadConnectome(args: argparse.Namespace = None) -> Dict[str, torch.tensor]:

    # if args is None:
    #     args = get_args()
    #     print(args)
    # plt.plot(range(10), range(10))
    # plt.show()
    #fmriPath="C:/Users/ghait/Repos/learningFromBrains/hcp/100307/analysis/100307_2_fsaverage_3T_rfMRI_REST1_LR-lh.stc"
    fmriPath ="C:/space_lin1/hcp/"
    #roiPath="C:/Users/ghait/Repos/learningFromBrains/hcp/100307/FreeSurfer/label/lh.aparc.a2009s.annot"
    #path='C:/Users/ghait/Repos/learningFromBrains/hcp/100307_3T_rfMRI_REST1_preproc/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz'
    #path='C:/Users/ghait/Repos/learningFromBrains/learning-from-brains/hcp/100307_3T_rfMRI_REST1_preproc/100307/T1w/Results/rfMRI_REST1_LR/PhaseOne_gdc_dc.nii.gz'
    #plotting.plot_img('C:/Users/ghait/Repos/learningFromBrains/learning-from-brains/hcp/100307_3T_rfMRI_REST1_preproc/100307/T1w/Results/rfMRI_REST1_LR/PhaseOne_gdc_dc.nii.gz')
    #plotting.plot_img('C:/Users/ghait/Repos/learningFromBrains/hcp/100307_3T_Structural_preproc/100307/T1w')
    #plotting.plot_img('C:/Users/ghait/Repos/learningFromBrains/hcp/100307_3T_Structural_unproc/100307/unprocessed/3T/T1w_MPR1/100307_3T_T1w_MPR1.nii.gz')

    #"C:/Users/ghait/Repos/learningFromBrains/hcp/100307_3T_Structural_unproc/100307/unprocessed/3T/T1w_MPR1/100307_3T_T1w_MPR1.nii.gz"
    #C:/Users/ghait/Repos/learningFromBrains/hcp/100307_3T_Structural_preproc/100307/T1w
#    print(nl.image.load_img(path).shape)
    #firstimg=nl.image.index_img(path,0)
    #print(firstimg.shape)
    #plotting.view_img(nl.image.mean_img(path), threshold=None)
    #fmriImg=nb.load(path)
    #fmriData=fmriImg.get_fdata()
    # sub=100307
    # task="REST1"
    # print("dirname")
    # print(basePath)
    # taskModifier='t'
    # if task=='REST':
    #     taskModifier='r'
    #directoryPath="C:/Users/ghait/Repos/learningFromBrains/hcp/100307/analysis/{}_2_fsaverage_3T_rfMRI_{}_LR-lh.stc".format(sub,task)
    #directoryPath=os.path.join(basePath, ("hcp/100307/analysis/{}_2_fsaverage_3T_rfMRI_{}_LR-lh.stc".format(sub,task)))
    # print(basePath)
    BasePath=basePath

    if "lin2" in BasePath:
        BasePath="/space_lin1"
    else:
        sub=100307
    # print('file path')
    # print( BasePath+"/hcp/{}/analysis/{}*{}*-lh.stc".format(sub,sub,task))

    directoryPathlhList = sorted(glob.glob(BasePath + "/hcp/{}/analysis/{}*{}*-lh.stc".format(sub, sub, task)))
    directoryPathrhList = sorted(glob.glob(BasePath + "/hcp/{}/analysis/{}*{}*-rh.stc".format(sub,sub,task)))
    if directoryPathlhList.__len__()==0 or directoryPathrhList.__len__() ==0:
        print('stc files not found')
        # return None


    directoryPathlh = directoryPathlhList[0]
    directoryPathrh = directoryPathrhList[0]

    # print('directoryPath')
    # print(directoryPath[0])
    # directoryPath=BasePath+"/hcp/{}/analysis/{}_2_fsaverage_3T_tfMRI_{}_LR-lh.stc".format(sub,sub,task)
    # # print(directoryPath)
    # # print(os.path.isfile(directoryPath))
    # if (not os.path.isfile(directoryPath)):
    #     directoryPath = BasePath + "/hcp/{}/analysis/{}_2_fsaverage_3T_rfMRI_{}_LR-lh.stc".format(sub, sub, task)
    #     # print(directoryPath)
    #     # print(os.path.isfile(directoryPath))
    # if (not os.path.isfile(directoryPath)):
    #     directoryPath = BasePath + "/hcp/{}/analysis/{}_2_fsaverage_rfMRI_{}_LR-lh.stc".format(sub, sub, task)
    #     # print(directoryPath)
    #     # print(os.path.isfile(directoryPath))
    # if (not os.path.isfile(directoryPath)):
    #     directoryPath = BasePath + "/hcp/{}/analysis/{}_2_fsaverage_tfMRI_{}_LR-lh.stc".format(sub, sub, task)
    #     # print(directoryPath)
    #     # print(os.path.isfile(directoryPath))
    fmriData = np.concatenate((mne.read_source_estimate(directoryPathlh).data,
                               mne.read_source_estimate(directoryPathrh).data),
                              axis=0)
    # fmriData=mne.read_source_estimate(directoryPath[0])


    # fileObj=open(directoryPath)
    #print(fileObj)
    #print("sep")
    # try:
    #  fmriData=mne.read_source_estimate(directoryPath)
    # except:
    #     directoryPath = basePath + "/hcp/{}/analysis/{}_2_fsaverage_{}fMRI_{}_LR-lh.stc".format(sub, sub,
    #                                                                                                taskModifier, task)
    #     fmriData = mne.read_source_estimate(directoryPath)

    #print(fmriData.shape)

    #roi=nb.freesurfer.io.read_annot(roiPath)
    #print(roi)
    #plotting.plot_stat_map(firstimg)

    # for img in nl.image.iter_img(path):
    #      #img is now an in-memory 3D img
    #     plotting.plot_stat_map(img, threshold=3, display_mode="z", cut_coords=1,
    #                            colorbar=False)
    #from nilearn.image import index_img
    #fmri_niimgs = nl.image.index_img(path)
    #print(fmri_niimgs.shape)
    #plotting.show()
    connectome= Connectome.connectome_batch(None,None,None,fmriData,sample)
    #print(connectome.shape)
    return torch.Tensor(connectome)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='nilearn testing')
    parser.add_argument(
        '--source-dir',
        metavar='DIR',
        type=str,
        help='path to HCP source directory'
    )
    parser.add_argument(
        '--target-dir',
        metavar='DIR',
        type=str,
        help='path where HCP data will be stored in BIDS format'
    )

    return parser.parse_args()


if __name__ == '__main__':
    loadConnectome()