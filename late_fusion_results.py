from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import glob
# from refuge-evaluation.util.file_management import sort_scores_by_filename

df_gt = pd.read_excel('~/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/dev_set_all/dev_set.xlsx')
y_true = df_gt.iloc[:, 1].values
y_filenames = df_gt.iloc[:, 0].values
print(y_true)

top_networks_ordered = ['DenseNet169','ResNet152V2','Xception','DenseNet201','EfficientNetB3','EfficientNetB0','InceptionV3','Resnet152','DenseNet121','MobileNetV2','NASNetLarge','VGG19','InceptionResNetV2','NASNetMobile']

best_performed_net = top_networks_ordered[0]

list_result_files = glob.glob("./results/*.xlsx")
classification_result_file = list(filter(lambda x: best_performed_net.lower() in x.lower(), list_result_files))[0]
df_gt = pd.read_excel(classification_result_file)
y_pred_net = df_gt.iloc[:, 1].values
np_result_array = np.array(y_pred_net)
np_result_array = np_result_array.reshape((-1, 1))

auc_max_fusion = []
aux_average_fusion = []

for index_net in range(1,len(top_networks_ordered)):
    # print(top_networks_ordered[index_net])
    classification_result_file = ''

    classification_result_file = list(filter(lambda x: top_networks_ordered[index_net].lower() in x.lower(), list_result_files))

    if len(classification_result_file) > 0:
        df_gt = pd.read_excel(classification_result_file[0])
        y_pred_net = df_gt.iloc[:, 1].values
        y_pred_net = y_pred_net.reshape((-1, 1))
        # print(np_result_array.shape)
        # print(y_pred_net.shape)
        # np.concatenate((a, b.T), axis=1)
        np_result_array = np.hstack((np_result_array, y_pred_net))

        max_results = np.max(np_result_array, axis=1)
        auc_max = roc_auc_score(y_true, max_results)
        auc_max_fusion.append(auc_max)

        average_results = np.mean(np_result_array, axis=1)
        auc_average = roc_auc_score(y_true, average_results)
        auc_max_fusion.append(auc_average)

        # print('Late fusion of top ' + str((index_net + 1)) + ' networks using Max probability : ' + str(auc_max))
        print('Late fusion of top ' + str((index_net + 1)) + ' networks using Average probability : ' + str(auc_average))
    else:
        print('No classification result file was found for ' + top_networks_ordered[index_net])




