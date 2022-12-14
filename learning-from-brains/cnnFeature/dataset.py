import configs
import glob
import tensorflow as tf
import PIL.Image as pil_image
import numpy as np
import mne


class Dataset(object):

    def __init__(self, path):
        self.imagesRef = sorted(glob.glob(path + '/*.stc'))
        #print("self.imagesRef")
        #print(self.imagesRef)

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

        fmriData = mne.read_source_estimate(self.imagesRef[idx]).data
        print(fmriData.shape)
        return fmriData

    def __len__(self):
        return len(self.imagesRef)


