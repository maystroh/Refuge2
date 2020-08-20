import io
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import tensorflow.keras as keras
# from PIL import Image
# from sklearn.metrics import roc_auc_score
# import PIL.Image
# import cv2
from config import config
from utils.dataset import get_dataset
from utils.custom_model import MySequential
from utils.custom_metrics import EuclideanDistanceMetric
import yaml
import math
import argparse

# Fixing seeds in order to fix this problem: https://stackoverflow.com/questions/48979426/keras-model-accuracy-differs-after-loading-the-same-saved-model
from numpy.random import seed

seed(42)  # keras seed fixing
tf.random.set_seed(42)  # tensorflow seed fixing

os.makedirs(config.SAVE_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)

def train(args):
    training_net = args.network
    batch_size = args.batch_size

    if 'EfficientNetB3' in training_net:
        IMG_SIZE = 320
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.EfficientNetB3(input_shape=IMG_SHAPE,
                                                          include_top=False,
                                                          weights='imagenet')

    if 'NASNetMobile' in training_net:
        IMG_SIZE = 224
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.NASNetMobile(input_shape=IMG_SHAPE,
                                                        include_top=False,
                                                        weights='imagenet')
    if 'DenseNet201' in training_net:
        IMG_SIZE = 224
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.densenet.DenseNet201(input_shape=IMG_SHAPE,
                                                                include_top=False,
                                                                weights='imagenet')
    if 'DenseNet121' in training_net:
        IMG_SIZE = 224
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.densenet.DenseNet121(input_shape=IMG_SHAPE,
                                                                include_top=False,
                                                                weights='imagenet')
    if 'DenseNet169' in training_net:
        IMG_SIZE = 224
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.densenet.DenseNet169(input_shape=IMG_SHAPE,
                                                                include_top=False,
                                                                weights='imagenet')

    if 'NASNetLarge' in training_net:
        batch_size = 6
        IMG_SIZE = 331
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.NASNetLarge(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
    if 'resnet152' in training_net:
        IMG_SIZE = 224
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.ResNet152(input_shape=IMG_SHAPE,
                                                     include_top=False,
                                                     weights='imagenet')

    if 'Xception' in training_net:
        IMG_SIZE = 299
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')

    if 'inception-resnet' in training_net:
        IMG_SIZE = 299
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,
                                                             include_top=False,
                                                             weights='imagenet')

    if 'resetNetV2' in training_net:
        IMG_SIZE = 224
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.ResNet152V2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
    if 'mobilenet_v2' in training_net:
        IMG_SIZE = 224
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

    if 'vgg19' in training_net:
        IMG_SIZE = 224
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights='imagenet')

    if 'EfficientNetB0' in training_net:
        IMG_SIZE = 224
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE,
                                                          include_top=False,
                                                          weights='imagenet')

    img_height = img_width = IMG_SIZE

    data_info_file = os.path.join(args.dataset_dir, 'dataset_info.yml')
    with open(data_info_file) as file:
        dataset_info = yaml.load(file, Loader=yaml.FullLoader)
        num_train_entries = dataset_info['train_size']
        num_val_entries = dataset_info['validation_size']
        regression = dataset_info['regression']
        num_labels = dataset_info['num_labels']
        dataset_info['img_width'] = img_width
        dataset_info['img_height'] = img_height
        dataset_info['zero_norm'] = args.normalization_zero
        # base_model.summary()
        last_layer = base_model.layers[-1]  # layer that you want to connect your new FC layer to
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(last_layer.output)
        if regression:  # regression
            new_top_layer = tf.keras.layers.Dense(num_labels)(global_average_layer)
        else:  # classification
            new_top_layer = tf.keras.layers.Dense(num_labels, activation='sigmoid')(global_average_layer)

        filepath = "./models" + "/saved-model-" + training_net + "-" + args.task_key + "-{epoch:02d}.hdf5"
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath, save_freq='epoch', verbose=1)

        logs = "logs/" + training_net + '-' + args.task_key + '-' + datetime.now().strftime("%Y%m%d-%H%M%S")
        config.file_writer = tf.summary.create_file_writer(logs)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs, write_graph=True,
                                                           update_freq='epoch')

        model = tf.keras.models.Model(base_model.input, new_top_layer)
        # model.summary()
        model = MySequential(model)
        model.trainable = True
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=args.lr),
                      loss=tf.keras.losses.BinaryCrossentropy() if not regression else tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy', tf.keras.metrics.AUC(num_thresholds=10000)] if not regression else EuclideanDistanceMetric(is_training=True))
        # model.summary()

        # one, get training and validation data
        train_ds = get_dataset(args.dataset_dir, 'train', batch_size, dataset_info, config.file_writer,
                               add_image_summaries=True)
        val_ds = get_dataset(args.dataset_dir, 'dev', batch_size, dataset_info, config.file_writer)

        history = model.fit_generator(train_ds,
                                      steps_per_epoch=math.ceil(num_train_entries / batch_size),
                                      epochs=args.epochs,
                                      validation_data=val_ds,
                                      validation_steps=math.ceil(num_val_entries / batch_size),
                                      callbacks=[model_checkpoint, tensorboard_callback])


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--dataset_dir', default='/home/hassan/ClusterGPU/data_GPU/hassan/REFUGE2/data/'
                                                       'refuge_data/datasets_fovea/',
                        help='directory containing the tf records')
    parser.add_argument('-o', '--network', default='DenseNet169',
                        help='The network architecture to use in the training')
    parser.add_argument('-tk', '--task_key', default='Localisation',
                        help='The key to use for generated files/folders')
    parser.add_argument('-ni', '--normalization_zero', action='store_true',
                        help='Zero normalization on batch of images')
    parser.add_argument('-n', '--batch_size', type=int, default=10,
                        help='batch size to update the weights')
    parser.add_argument('-ep', '--epochs', type=int, default=100,
                        help='Number of epoch to train the dataset')
    parser.add_argument('-lrate', '--lr', type=float, default=0.00001,
                        help='Number of epoch to train the dataset')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
