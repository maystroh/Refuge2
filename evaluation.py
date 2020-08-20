# make a prediction for a new image.
# from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import xlwt
import os
from utils.image_processing import preprocess_image
from utils.custom_model import MySequential
from utils.custom_metrics import EuclideanDistanceMetric

# Fixing seeds in order to fix this problem: https://stackoverflow.com/questions/48979426/keras-model-accuracy-differs-after-loading-the-same-saved-model
from numpy.random import seed
import numpy as np
import pandas as pd

seed(42)  # keras seed fixing
tf.random.set_seed(42)  # tensorflow seed fixing

img_height = 224
img_width = 224

model = load_model('./models/new_models/saved-model-efficient-originalImage-03.hdf5',
                   custom_objects={'MySequential': MySequential,
                                   'EuclideanDistanceMetric': EuclideanDistanceMetric(is_training=False)})


# load an image and predict the class
def run_example(isfovea_loc=False, refuge_validation=False):

    if not isfovea_loc:
        sheet1.write(0, 0, 'FileName')
        sheet1.write(0, 1, 'Glaucoma Risk')
    else:
        sheet1.write(0, 0, 'ImageName')
        sheet1.write(0, 1, 'Fovea_X')
        sheet1.write(0, 2, 'Fovea_Y')
    i = 1
    # load the image
    if refuge_validation:
        file_path = "/home/hassan/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/Refuge2-Validation"
        file_list = os.listdir(file_path)
    else:
        file_path = "/home/hassan/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/dataset_classification"
        dataframe = pd.read_excel(os.path.join(file_path, 'dev_set.xlsx'))
        file_list = dataframe['FileName'].values

    file_list.sort()
    for fileName in file_list:
        # print(fileName)
        if refuge_validation:
            newFileName = fileName[0:fileName.find(".")] + '.jpg'
            # print(newFileName)
            sheet1.write(i, 0, newFileName)
        else:
            sheet1.write(i, 0, fileName)
        path = os.path.join(file_path, "%s" % fileName)

        # img = load_image(path)
        img = load_img(path)
        img = img_to_array(img)
        factor_width = img.shape[0] / img_width
        factor_height = img.shape[1] / img_height
        factors_tensor = [factor_width, factor_height]

        img = preprocess_image(img, img.shape[0], img.shape[1], img_width, img_height, is_training=False,
                               add_image_summaries=False)

        img = np.expand_dims(img, axis=0)

        result = model.predict(img)
        if not isfovea_loc:
            print(path + ' -> ' + str(result[0][0]))
            sheet1.write(i, 1, float(result[0][0]))
        else:
            print(path + ' -> ' + str(result[0] * factors_tensor))
            sheet1.write(i, 1, float(result[0][0]) * factors_tensor[0])
            sheet1.write(i, 2, float(result[0][1]) * factors_tensor[1])

        i = i + 1


workbook = xlwt.Workbook(encoding='utf-8')
sheet1 = workbook.add_sheet("Classification")
run_example(isfovea_loc=False, refuge_validation=False)
results_file_path = './results/classification_results_DenseNet169.xlsx'
# results_file_path = './results/localisation_results.xlsx'

workbook.save(results_file_path)

import pandas as pd

data_xls = pd.read_excel(results_file_path, index_col=0)
# data_xls.to_csv('./results/fovea_location_results_resnet152.csv', encoding='utf-8')
data_xls.to_csv('./results/classification_results.csv', encoding='utf-8')

# command to launch to evaluate the results
# python3 ./refuge-evaluation/evaluate_single_submission.py ./results ~/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/REFUGE-Test400/REFUGE-Test-GT --output_path ./output --export_table true
# python3 ./refuge-evaluation/evaluate_single_submission.py ./results ~/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/dataset_classification/dev_set.xlsx --output_path ./output --export_table true
