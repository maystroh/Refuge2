"""dataset.py
This module implements functions for reading ImageNet (ILSVRC2012)
dataset in TFRecords format.
"""

import os
from functools import partial

import tensorflow as tf
import numpy as np
from config import config
from utils.image_processing import preprocess_image, resize_and_rescale_image


def decode_jpeg(image_buffer, image_format, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(name=scope):
        switcher_format = {
            'jpg': tf.image.decode_jpeg,
            'jpeg': tf.image.decode_jpeg,
            'png': tf.image.decode_png,
            'bmp': tf.image.decode_bmp,
        }

        # Decode the string as an RGB .
        # Note that the resulting image contains an unknown height
        # and width that is set dynamically by decode. In other
        # words, the height and width of image is unknown at compile time.
        image = switcher_format[image_format](image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).
        # The various adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def _parse_fn(example_serialized, is_training, num_labels, img_width, img_height, regression, normalization, tb_writer,
              add_image_summaries=False, troubleshooting=False):
    """Helper function for parse_fn_train() and parse_fn_valid()
    Each Example proto (TFRecord) contains the following fields:

    Args:
        example_serialized: scalar Tensor tf.string containing a serialized Example protocol buffer.
        is_training: training (True) or validation (False).
    Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
    """
    image_feature_description = {
        'image_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'image_url': tf.io.FixedLenFeature([], dtype=tf.string),
        'labels': tf.io.FixedLenFeature([num_labels], tf.float32,
                                        default_value=tf.zeros([num_labels], dtype=tf.float32)) if regression
        else tf.io.FixedLenFeature([num_labels], tf.int64, default_value=tf.zeros([num_labels], dtype=tf.int64)),

        'height': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'width': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1)
    }

    feature_parsed = tf.io.parse_single_example(example_serialized, image_feature_description)
    image = decode_jpeg(feature_parsed['image_raw'], 'jpg', scope='Decode_jpg')

    imageurl = feature_parsed['image_url']
    labels = feature_parsed['labels']
    height = feature_parsed['height']
    width = feature_parsed['width']

    if config.DATA_AUGMENTATION:
        image = preprocess_image(image, height, width, img_width, img_height, is_training=is_training)
    else:
        image = resize_and_rescale_image(image, img_width, img_height, do_mean_subtraction=normalization)

    # For fovea detection..
    factor_width = tf.cast(width / img_width, tf.float32)
    factor_height = tf.cast(height / img_height, tf.float32)
    factors_tensor = tf.stack([factor_width, factor_height], axis=0)
    print(labels)
    print(factors_tensor)
    labels_resized = tf.divide(labels, factors_tensor)

    # if troubleshooting:
    #     return image, imageurl, labels, labels_resized, factors_tensor, height, width,factor_width, factor_height
    #
    return image, labels_resized

    # return image, imageurl, labels, height, width
    # return image, labels


def get_dataset(tfrecords_dir, subset, batch_size, dataset_info, tb_writer=None, add_image_summaries=False,
                troubleshooting=False):
    """Read TFRecords files and turn them into a TFRecordDatasets."""
    files = tf.io.matching_files(os.path.join(tfrecords_dir, 'dataset_%s_*' % subset))
    shards = tf.data.Dataset.from_tensor_slices(files)
    # shards = shards.repeat()
    shards = shards.shuffle(buffer_size=dataset_info['train_size'] // 10)

    dataset = shards.interleave(tf.data.TFRecordDataset, num_parallel_calls=config.NUM_DATA_WORKERS)  # Parallelize

    # this function is to apply a function on each item in the dataset
    parser = partial(_parse_fn, is_training=True if subset == 'train' else False, num_labels=dataset_info['num_labels'],
                     img_width=dataset_info['img_width'], img_height=dataset_info['img_height'],
                     regression=dataset_info['regression'], normalization=dataset_info['zero_norm'],
                     tb_writer=tb_writer,
                     add_image_summaries=add_image_summaries, troubleshooting=troubleshooting)
    dataset = dataset.map(map_func=parser, num_parallel_calls=config.NUM_DATA_WORKERS)  # Parallelize map transformation
    dataset = dataset.batch(batch_size=batch_size)

    dataset = dataset.cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
