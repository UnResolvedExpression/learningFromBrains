import configs
import glob
import tensorflow as tf
import PIL.Image as pil_image
import numpy as np
import mne
import nibabel as nb

class Dataset(object):

    def __init__(self, path):
        #/sub-02/ses-a1/func/sub-02_ses-a1_task-a_run-1_bold.nii.gz

        self.imagesRef = sorted(glob.glob(path + '/*/Preprocessed/*.nii.gz'))
        self.imagesRef=self.imagesRef[0:3] #for a quick test
        #print("self.imagesRef")
        #print(self.imagesRef)
        # for f_a in glob.glob(path + '/*/analysis/*-lh.stc'):
        #     # process_file(f_a, file_type='a')
        #     # process_file(file_directory + f_a[:-11] + "_data_b.dat", file_type='b')
        #     self.imagesRef+= (f_a),(path + f_a.replace('lh','rh'))
    def __getitem__(self, idx):
        # image = tf.io.read_file(self.imagesRef[idx])
        # #print("image")
        # #print(image)
        #
        # image = tf.image.decode_jpeg(image, channels=3)
        # image = tf.image.resize_with_pad(image, 100,100)
        # #image = pil_image.fromarray(image.numpy())
        # image = np.array(image).astype(np.float32)
        # image = np.transpose(image, axes=[2, 0, 1])
        # # normalization
        # #image /= 255.0
        # #print(image)
        # lh,rh=self.imagesRef[idx]
        # lh=self.imagesRef[idx]
        # rh=lh.replace('lh','rh')
        # print('mne.read_source_estimate(lh).data')
        # print(mne.read_source_estimate(lh).data)
        # print(mne.read_source_estimate(lh).data.shape)
        # fmriData = np.concatenate((mne.read_source_estimate(lh).data,
        #                            mne.read_source_estimate(rh).data),
        #                           axis=0)
        #fmriData=mne.read_source_estimate(rh).data
        # print(fmriData.shape)
        fmriData=nb.load(self.imagesRef[idx]).get_fdata()
        return fmriData

    def __len__(self):
        return len(self.imagesRef)


