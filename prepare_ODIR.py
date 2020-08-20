import pandas as pd
import glob
import shutil

train_gt_files = '~/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/dataset_classification/ODIR-5K_Training_Images/ODIR-5K_Training_Annotations(Updated)_V2.xlsx'

dataset_df = pd.read_excel(train_gt_files)
total_num = len(dataset_df.index)
print(total_num)

left_eye_normal = dataset_df[~dataset_df['Left-Diagnostic Keywords'].str.contains("glaucoma")]

left_eye_glaucoma = dataset_df[dataset_df['Left-Diagnostic Keywords'].str.contains("glaucoma")]
left_eye_num_glau = len(left_eye_glaucoma.index)
left_eye_num_normal = total_num - left_eye_num_glau
print(left_eye_num_glau)
print(len(left_eye_normal.index))

right_eye_normal = dataset_df[~dataset_df['Right-Diagnostic Keywords'].str.contains("glaucoma")]
right_eye_glaucoma = dataset_df[dataset_df['Right-Diagnostic Keywords'].str.contains("glaucoma")]
right_eye_num_glau = len(right_eye_glaucoma.index)
right_eye_num_normal = total_num - right_eye_num_glau
print(right_eye_num_glau)
print(left_eye_num_normal)

percentage_dev = 20  # in percentage

right_eye_num_glau_dev = int(percentage_dev / 100. * right_eye_num_glau)
right_eye_num_normal_dev = int(percentage_dev / 100. * right_eye_num_normal)
left_eye_num_glau_dev = int(percentage_dev / 100. * left_eye_num_glau)
left_eye_num_normal_dev = int(percentage_dev / 100. * left_eye_num_normal)

df_train = pd.DataFrame(columns=['FileName', 'Glaucoma Risk'])
df_dev = pd.DataFrame(columns=['FileName', 'Glaucoma Risk'])

nb_right_eye_glau = 1
nb_left_eye_glau = 1

nb_right_eye_normal = 1
nb_left_eye_normal = 1

initial_path = '/home/hassan/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/dataset_classification/'

import csv

odir_file = open('/home/hassan/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/dataset_classification/train_ODIR'
                 '.csv', 'w', newline='')
writer_train = csv.writer(odir_file)
odir_file_dev = open('/home/hassan/ClusterGPU/data_GPU/hassan/REFUGE2/data/refuge_data/dataset_classification'
                     '/dev_ODIR.csv', 'w', newline='')
writer_dev = csv.writer(odir_file_dev)

dest_train_dir = initial_path + 'train/ODIR/'
dest_dev_dir = initial_path + 'dev/ODIR/'

initial_path_images = initial_path + 'ODIR-5K_Training_Images/ODIR-5K_Training_Dataset/'

for index, row in left_eye_glaucoma.iterrows():
    image_path = initial_path_images + row['Left-Fundus']
    if nb_left_eye_glau <= left_eye_num_glau_dev:
        shutil.copy(image_path, dest_dev_dir)
        print('copy to %s/%s ' % (dest_dev_dir, row['Left-Fundus']))
        writer_dev.writerow(['./dev/ODIR/%s' % row['Left-Fundus'], 1])
        nb_left_eye_glau = nb_left_eye_glau + 1
    else:
        shutil.copy(image_path, dest_train_dir)
        writer_train.writerow(['./train/ODIR/%s' % row['Left-Fundus'], 1])
        print('copy %s/%s' % (dest_train_dir, row['Left-Fundus']))

for index, row in left_eye_normal.iterrows():
    image_path = initial_path_images + row['Left-Fundus']
    if nb_left_eye_normal <= left_eye_num_normal_dev:
        shutil.copy(image_path, dest_dev_dir)
        print('copy %s/%s ' % (dest_dev_dir, row['Left-Fundus']))
        writer_dev.writerow(['./dev/ODIR/%s' % row['Left-Fundus'], 0])
        nb_left_eye_normal = nb_left_eye_normal + 1
    else:
        shutil.copy(image_path, dest_train_dir)
        writer_train.writerow(['./train/ODIR/%s' % row['Left-Fundus'], 0])
        print('copy %s/%s' % (dest_train_dir, row['Left-Fundus']))

for index, row in right_eye_glaucoma.iterrows():
    image_path = initial_path_images + row['Right-Fundus']
    if nb_right_eye_glau <= right_eye_num_glau_dev:
        shutil.copy(image_path, dest_dev_dir)
        print('copy %s/%s ' % (dest_dev_dir, row['Right-Fundus']))
        nb_right_eye_glau = nb_right_eye_glau + 1
        writer_dev.writerow(['./dev/ODIR/%s' % row['Right-Fundus'], 1])
    else:
        shutil.copy(image_path, dest_train_dir)
        writer_train.writerow(['./train/ODIR/%s' % row['Right-Fundus'], 1])
        print('copy %s/%s ' % (dest_train_dir, row['Right-Fundus']))

for index, row in right_eye_normal.iterrows():
    image_path = initial_path_images + row['Right-Fundus']
    if nb_right_eye_normal <= right_eye_num_normal_dev:
        shutil.copy(image_path, dest_dev_dir)
        print('copy %s/%s ' % (dest_dev_dir, row['Right-Fundus']))
        writer_dev.writerow(['./dev/ODIR/%s' % row['Right-Fundus'], 0])
        nb_right_eye_normal = nb_right_eye_normal + 1
    else:
        shutil.copy(image_path, dest_train_dir)
        writer_train.writerow(['./train/ODIR/%s' % row['Right-Fundus'], 0])
        print('copy %s/%s ' % (dest_train_dir, row['Right-Fundus']))

# df.loc[i] = ['name' + str(i)] + list(randint(10, size=2))
