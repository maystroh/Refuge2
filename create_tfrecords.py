from random import shuffle
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import math
import argparse
import os
import random
from config import config


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def __write_labels(output_directory,
                   num_labels,
                   label_names,
                   file_name):
    label_file_name = os.path.join(output_directory, file_name)
    with tf.io.gfile.GFile(label_file_name, 'w') as f:
        for label_index, label_name in zip(range(num_labels), label_names):
            f.write('{}:{}\n'.format(label_index, label_name))


def __write_dataset_info(output_directory,
                         num_labels,
                         num_train_entries,
                         image_format,
                         num_validation_entries,
                         regression):
    with open(os.path.join(output_directory, 'dataset_info.yml'), 'w') as f:
        f.write('num_labels: {}\n'.format(num_labels))
        f.write('train_size: {}\n'.format(num_train_entries))
        f.write('validation_size: {}\n'.format(num_validation_entries))
        f.write('image_format: {}\n'.format(image_format))
        f.write('regression: {}\n'.format(1 if regression else 0))


def __process_image(filename):
    """Process a single image file.

    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = tf.image.decode_jpeg(image_data, channels=3)
    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def __create__data_record(output_path, images_directory, split_name, num_shards, filenames, image_labels, regression):
    # number of entries per binary file
    num_entries = len(filenames)
    num_per_shard = int(math.ceil(num_entries / float(num_shards)))

    num_saved_entries = 0
    extension_entries = ''

    for shard_ind in range(num_shards):

        output_file_name = config.FILE_PATTERN % (split_name, shard_ind + 1, num_shards)
        out_filename = os.path.join(output_path, output_file_name)
        # open the TFRecords file
        writer = tf.io.TFRecordWriter(out_filename)

        start_ndx = shard_ind * num_per_shard
        end_ndx = min((shard_ind + 1) * num_per_shard, num_entries)

        for i in range(start_ndx, end_ndx):
            print(filenames[i])
            # Load the image
            file_name, extension = os.path.splitext(filenames[i])
            if extension_entries == '' and extension[1:] not in extension_entries:
                extension_entries = extension[1:]
            elif extension[1:] not in extension_entries:
                extension_entries = ',' + extension[1:]

            image_path = images_directory + filenames[i]
            image_data, height, width = __process_image(image_path)

            labels = image_labels[i].tolist()

            if image_data is not None:
                # Create a feature
                feature = {
                    'image_raw': _bytes_feature(image_data),
                    'image_url': _bytes_feature(filenames[i].encode('utf-8')),
                    'labels': _float64_feature(labels) if regression else _int64_feature(labels),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width)
                }
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
                num_saved_entries = num_saved_entries + 1
            else:
                print('Image {} could not be read!'.format(filenames[i]))

        writer.close()
        sys.stdout.flush()

    return num_saved_entries, extension_entries


def create_shards(args):
    train_gt_files = glob.glob(args.images_directory + '*train*.xlsx')
    dev_gt_files = glob.glob(args.images_directory + '*dev*.xlsx')

    assert len(train_gt_files) > 0 or len(dev_gt_files) > 0, "Ground truth files do not exist in the provided path"

    train_gt_path = train_gt_files[0]
    val_gt_path = dev_gt_files[0]

    train_df = pd.read_excel(train_gt_path)
    label_names = train_df.columns.values[1:]

    train_image_filenames = train_df.iloc[:, 0].values
    train_images_labels = train_df.iloc[:, 1:].values

    val_df = pd.read_excel(val_gt_path)
    val_image_filenames = val_df.iloc[:, 0].values
    val_images_labels = val_df.iloc[:, 1:].values

    print('Creating training tfrecords')
    num_train_entries, format_entries = __create__data_record(args.output_directory, args.images_directory, 'train',
                                                              args.num_shards,
                                                              train_image_filenames, train_images_labels,
                                                              args.regression)
    print('Creating development tfrecords')
    num_dev_entries, _ = __create__data_record(args.output_directory, args.images_directory, 'dev', args.num_shards,
                                               val_image_filenames, val_images_labels, args.regression)

    num_labels = len(label_names)

    # Writing dataset information
    __write_dataset_info(args.output_directory,
                         num_labels,
                         num_train_entries,
                         format_entries,
                         num_dev_entries,
                         args.regression)
    __write_labels(args.output_directory,
                   num_labels,
                   label_names,
                   config.LABEL_FILE_NAME)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--images_directory', default='/home/hassan/ClusterGPU/data_GPU/hassan'
                                                            '/REFUGE2/data/refuge_data/dataset_classification/',
                        help='directory containing the ground truth files (excel or csv)')
    parser.add_argument('-o', '--output_directory', default='/home/hassan/ClusterGPU/data_GPU/hassan'
                                                            '/REFUGE2/data/refuge_data/dataset_classification/',
                        help='directory where the TF records should be generated')
    parser.add_argument('-n', '--num_shards', type=int, default=3,
                        help='number of output binary files per dataset split')
    parser.add_argument('--regression', dest='regression', action='store_true',
                        help='format targets for a regression problem')
    parser.add_argument('--classification', dest='classification', action='store_true',
                        help='format targets for a classification problem')

    args = parser.parse_args()

    if args.classification and args.regression:
        raise ValueError('cannot set both classification and regression')

    create_shards(args)


if __name__ == '__main__':
    main()
