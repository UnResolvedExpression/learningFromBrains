#!/usr/bin/env python3

import os
import numpy as np
from typing import Tuple
import re
import glob
from dirname import basePath


def grab_tarfile_paths(path) -> Tuple[str]:
    paths = os.listdir(path)
    # relevant_path = "[path to folder]"
    included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
    # included_tasks= ['REST1']
    #included_tasks= ['REST1','EMOTION','SOCIAL','WM']
    # paths = [fn for fn in paths
    #               if any(task in fn for task in included_tasks)]

    # str = re.split(r'-|_', sample["__key__"])
    # (str[3], str[5], str[7], str[9])
    # # here we will remove entries we do not have the files for
    # directoryPathlhList = sorted(glob.glob(BasePath + "/hcp/{}/analysis/{}*{}*-lh.stc".format(sub, sub, task)))
    # directoryPathrhList = sorted(glob.glob(BasePath + "/hcp/{}/analysis/{}*{}*-rh.stc".format(sub, sub, task)))
    # if directoryPathlhList.__len__() or directoryPathrhList.__len__() == 0:
    #     continue

    for p in paths:
        str = re.split(r'-|_', p)
        print('path')
        print(p)
        print(str)
        (sub,task)=(str[3], str[5])
        BasePath=basePath
        if "lin2" in BasePath:
            BasePath = "/space_lin1"
        directoryPathlhList = sorted(glob.glob(BasePath + "/hcp/{}/analysis/{}*{}*-lh.stc".format(sub, sub, task)))
        directoryPathrhList = sorted(glob.glob(BasePath + "/hcp/{}/analysis/{}*{}*-rh.stc".format(sub, sub, task)))
        print('directoryPathlhList')
        print(BasePath + "/hcp/{}/analysis/{}*{}*-lh.stc".format(sub, sub, task))
        print(directoryPathlhList)
        print(directoryPathrhList)
        print(len(directoryPathlhList)==0)
        print(len(directoryPathrhList)==0)

        if len(directoryPathlhList)==0 or len(directoryPathrhList)==0:
            paths.remove(path)
            print('removed')
    print('len(paths)')
    print(len(paths))
    tarfiles = []


    for p in paths:

        if os.path.isdir(
            os.path.join(
                path,
                p
            )
        ):
            tarfiles += [
                os.path.join(
                    path,
                    p,
                    f
                )
                for f in os.listdir(
                    os.path.join(
                        path,
                        p
                    )
                )
                if f.endswith('.tar')
            ]

        elif np.logical_and(
            os.path.isfile(
                os.path.join(
                    path,
                    p
                )
            ), 
            p.endswith('.tar')
        ):
            tarfiles.append(
                os.path.join(
                    path,
                    p
                )
            )
    print('len(tarfiles)')
    print(len(tarfiles))
    return sorted(np.unique(tarfiles))


def split_tarfile_paths_train_val(
    tarfile_paths,
    frac_val_per_dataset: float=0.05,
    n_val_subjects_per_dataset: int=None,
    n_test_subjects_per_dataset: int=None,
    n_train_subjects_per_dataset: int=None,
    min_val_per_dataset: int=2,
    seed: int=1234
    ) -> Tuple[str]:
    np.random.seed(seed)
    datasets = np.unique(
        [
            f.split('/')[-1].split('ds-')[1].split('_')[0]
            for f in tarfile_paths
        ]
    )
    train_tarfiles, val_tarfiles = [], []
    test_tarfiles = [] if n_test_subjects_per_dataset is not None else None
    
    for dataset in datasets:
        dataset_tarfiles = np.unique(
            [
                f for f in tarfile_paths
                if f'ds-{dataset}' in f
            ]
        )

        if n_val_subjects_per_dataset is None and \
           n_test_subjects_per_dataset is None and \
           n_train_subjects_per_dataset is None:
            np.random.shuffle(dataset_tarfiles)
            n_val = max(
                int(len(dataset_tarfiles)*frac_val_per_dataset),
                min_val_per_dataset
            )
            train_tarfiles += list(dataset_tarfiles[:-n_val])
            val_tarfiles += list(dataset_tarfiles[-n_val:])

        else:
            subjects = np.unique(
                [
                    f.split('_sub-')[1].split('_')[0]
                    for f in dataset_tarfiles
                ]
            )
            n_test_subjects_per_dataset = 0 if n_test_subjects_per_dataset is None else n_test_subjects_per_dataset
            assert n_val_subjects_per_dataset is not None,\
                'n_train_subjects_per_dataset and n_val_subjects_per_dataset must be specified'
            assert n_val_subjects_per_dataset < len(subjects),\
                'n_val_subjects_per_dataset must be smaller than the number of subjects'
            n_train_subjects_per_dataset = len(subjects)-n_val_subjects_per_dataset if n_train_subjects_per_dataset is None else n_train_subjects_per_dataset
            assert (
                n_val_subjects_per_dataset+\
                n_test_subjects_per_dataset+\
                n_train_subjects_per_dataset
            ) <= len(subjects), \
                f'Not enough subjects in dataset {dataset} for '\
                f'{n_val_subjects_per_dataset} val, '\
                f'{n_test_subjects_per_dataset} test, '\
                f'{n_train_subjects_per_dataset} train'

            validation_subjects = np.random.choice(
                subjects,
                n_val_subjects_per_dataset,
                replace=False
            )
            if n_test_subjects_per_dataset > 0:
                test_subjects = np.random.choice(
                    [s for s in subjects if s not in validation_subjects],
                    n_test_subjects_per_dataset,
                    replace=False
                )
            else:
                test_subjects = []

            train_subjects = [
                s for s in subjects
                if s not in validation_subjects
                and s not in test_subjects
            ][:n_train_subjects_per_dataset]

            for subject in subjects:
                
                if subject in validation_subjects:
                    val_tarfiles += [
                        f for f in dataset_tarfiles
                        if f'sub-{subject}' in f
                    ]
                
                elif subject in train_subjects:
                    train_tarfiles += [
                        f for f in dataset_tarfiles
                        if f'sub-{subject}' in f
                    ]
                
                elif subject in test_subjects:
                    test_tarfiles += [
                        f for f in dataset_tarfiles
                        if f'sub-{subject}' in f
                    ]

                else:
                    continue
    
    if test_tarfiles is None:
        return {
            'train': train_tarfiles,
            'validation': val_tarfiles,
        }
    
    else:
        return {
            'train': train_tarfiles,
            'validation': val_tarfiles,
            'test': test_tarfiles
        }