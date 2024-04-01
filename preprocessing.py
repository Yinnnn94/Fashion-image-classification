import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image 
import numpy as np
import pickle
# import tensorflow as tf


data = pd.read_csv(r'C:\Users\user\Downloads\myntradataset\styles.csv', on_bad_lines = 'skip', usecols = ['id', 'gender', 'masterCategory', 'baseColour', 'season', 'usage'])
data = data.dropna()


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"路径 '{path}' 不存在，已創建成功。")
    else:
        print(f"路径 '{path}' 已存在。")
    return path

def sep_files(path, data): # 將照片依照不同的條件分類並寫入至txt中
    filter_col = ['baseColour', 'season', 'usage', 'masterCategory', 'gender']
    for i in filter_col:
        storage_path = create_path_if_not_exists(f'{path}\{i}')
        filter = data[i].unique()
        for f in filter:
            with open(f'{storage_path}\\{f}.txt', 'w') as file:
                for id_ in data[data[i] == f]['id']:
                    file.write(f"C:\\Users\\user\\Downloads\\myntradataset\\images\\{str(id_)}.jpg") 
                    file.write('\n')

def move_files(path, data): # 將照片依照不同的條件分類並放入至不同的資料夾中
    filter_col = ['season', 'usage', 'masterCategory', 'gender']
    for i in filter_col:
        storage_path = create_path_if_not_exists(f'{path}\{i}')
        filter = data[i].unique()
        for f in filter:
            create_path_if_not_exists(f'{storage_path}\{f}')
            for id_ in data[data[i] == f]['id']:
                if str(id_)+ '.jpg' in os.listdir(r"C:\Users\user\Downloads\myntradataset\images"):
                    os.rename(f'C:\\Users\\user\\Downloads\\myntradataset\\images\\{str(id_)}.jpg', f'{storage_path}/{f}/{str(id_)}.jpg')
                else:
                   print(f'{id_}不存在')


root = r'C:\Users\user\Downloads\images'

image_path = r"C:\Users\user\Downloads\myntradataset\images"



# move_files(path, data)
# sep_files(path, data)
def save_array_to_pickle(filter_col, array):
    with open(f"{filter_col}.pkl", 'wb') as f:
        pickle.dump(array, f)

# 將各別照片轉成array，且不先存至pickle中
def img_to_array(path, filter_col, data):
    img_array = []
    img_label = []
    filter_data = data[['id', filter_col]]
    filter_name = filter_data[filter_col].unique()
    filter_label = {name: i for i, name in enumerate(filter_name)}
    for i, key in enumerate(filter_label):
        print(i, key)
        for file in (filter_data[filter_data[filter_col] == key]['id'][:3]):
            if str(file) + '.jpg' not in os.listdir(path):
                print(f'{file}不存在')
            else:
                # print(file)
                jpg_path = f'{path}\{str(file)}.jpg'
                images = plt.imread(jpg_path)
                img_array.append(images)
                    # pickle.dump(image, f)
                    # # img_label.append(i)




        # print(img_array, len(img_array), img_label, len(img_label))
        # return img_array, img_label 
# gender_array, gender_label = img_to_array(image_path, 'gender', data)
img_to_array(image_path, 'gender', data)

# print(gender_array, gender_label, len(gender_array), len(gender_label))
# gender_class = ['Boys', 'Girls', 'Men', 'Unisex', 'Women']
# gender_label = {name: i for i, name in enumerate(gender_class)} # 產生出各個class的label (Dictionary)

# for g in gender_class:
#     img_array = []
#     with open(f'{path}\gender\{g}.txt', 'r') as file: # 讀取檔案
#         files = file.readlines()
#         for f in files:
#             image = img.imread(f[:-1])
#             img_array.append(image)
#     print(g, len(img_array))

# img_to_array(r'C:\Users\user\OneDrive\文件\Python Scripts\Fashion_Classification\images\gender\Boys.txt')
# def plot(path):
#     fig, axs = plt.subplots(1, 5)
#     with open(path, 'r') as file:
#         files = file.readlines()
#     for i, ax in zip(files[:5], axs.flat):
#         image = img.imread(i[:-1])
#         ax.imshow(image, cmap = 'gray')
#     plt.show()

# def CNN_model(train_data, test_data, epochs = 10):
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')])
#     model.compile(optimizer ='adam', loss ='crossentropy', metrics = ['accuracy'])
#     history = model.fit(train_data, test_data, epochs = epochs)
#     return model, history


