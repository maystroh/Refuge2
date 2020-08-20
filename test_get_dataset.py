"""test_get_dataset.py

This test script could be used to verify either the 'train' or
'validation' dataset, by visualizing data augmented images on
TensorBoard.

Examples:
$ cd ${HOME}/project/keras_imagenet
$ python3 test_get_dataset.py train
$ tensorboard --logdir logs/train
"""

import os
import shutil
import argparse
import yaml
import tensorflow as tf
import numpy as np
from PIL import Image

from utils.dataset import get_dataset

# dataset_dir = '/home/hassan/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/datasets_fovea'
dataset_dir = '/home/hassan/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/dataset_classification'


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-s', '--subset', type=str, default='train')
args = parser.parse_args()

# log_dir = os.path.join('logs', args.subset)
# shutil.rmtree(log_dir, ignore_errors=True)  # clear prior log data

data_info_file = os.path.join(dataset_dir, 'dataset_info.yml')
num_train_entries = 0
num_val_entries = 0
file = open(data_info_file)
dataset_info = yaml.load(file, Loader=yaml.FullLoader)
dataset_info['img_width'] = 224
dataset_info['img_height'] = 224
dataset_info['zero_norm'] = False
dataset = get_dataset(dataset_dir, args.subset, batch_size=1, dataset_info=dataset_info, troubleshooting=True)


from tensorflow.keras.models import load_model
from utils.custom_model import MySequential

model = load_model('./models/saved-model-EfficientNetB0-localisation-21.hdf5',
                   custom_objects={'MySequential': MySequential})

for next_element in dataset:
    images = next_element[0]

    np_images = tf.image.convert_image_dtype(images, dtype=tf.uint8, saturate=True)
    print(np_images.shape)

    np_images = tf.make_ndarray(tf.make_tensor_proto(np_images))
    np_images = np.squeeze(np_images, 0)
    gr_im = Image.fromarray(np_images).save('test.png')
    print(np_images.shape)
    np_image_standardized = tf.image.per_image_standardization(np_images)
    np_image_stan = tf.make_ndarray(tf.make_tensor_proto(np_image_standardized))
    print(np_image_stan.shape)
    gr_im = Image.fromarray(np_image_stan).save('test_stan.png')


    image_url = next_element[1]
    labels = next_element[2]
    labels_resized = next_element[3]
    factors = next_element[4]
    height = next_element[5]
    width = next_element[6]

    np_images = np_images.reshape(1, 224, 224, 3)
    print(np_images.shape)
    # result = model.predict(np_images)
    # print(result)

    tf.print(image_url)
    tf.print(labels)
    tf.print(height)
    tf.print(width)

writer.close()
