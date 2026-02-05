import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import tensorflow as tf
import os

def preprocess_mri(file_path, target_shape=(128, 128, 128)):
    
    input = []
    image_data_resized = np.load(os.path.join(file_path))
    input.append(image_data_resized)
    input = tf.image.resize(input, (64, 64)).numpy()

    return input
