import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

start_time = time.time()
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


def save_array_to_pickle(filter_col, array):
    with open(f"{filter_col}.pkl", 'wb') as f:
        pickle.dump(array, f)

# 將各別照片轉成array，且不先存至pickle中
def img_to_array(path, filter_col, data):
    img_array = []
    img_label_counts = []
    filter_data = data[['id', filter_col]]
    filter_name = filter_data[filter_col].unique()
    filter_label = {name: i for i, name in enumerate(filter_name)}
    for i, key in enumerate(filter_label):
        print(i, key)
        for file in (filter_data[filter_data[filter_col] == key]['id'][:50]):
            if str(file) + '.jpg' not in os.listdir(path):
                print(f'{file}不存在')
            else:
                jpg_path = f'{path}\{str(file)}.jpg'
                images = plt.imread(jpg_path)
                img_array.append(images.astype('float32')/255)
                img_label_counts.append(i)

    return img_array, img_label_counts , filter_label

gender_array, gender_label_counts, gender_label = img_to_array(image_path, 'gender', data)


##### 必須將list裡的資料轉乘numpy!!!!!!!!!!!!!!!!!!

# 將label的資料計數
def label_count_plot(label):
    value = list(set(label))
    value_counts = [label.count(v) for v in value]
    print(value_counts)
    plt.bar(value, value_counts)
    plt.show()

# label_count_plot(gender_label_counts)

# 繪製圖片
def plot_img(array):
    for i in array:
        plt.imshow(i)
        plt.show()

x_train, x_test, y_train, y_test = train_test_split(gender_array, gender_label_counts)

print(x_train[0], x_train[0].shape, len(x_train))
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3),
                 activation='relu',
                 input_shape = (80, 60, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.summary()
model.compile(optimizer ='adam', loss ='categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 10, batch_size = 10)
# print(history.history['accuracy'].shape)

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()

end_time = time.time()
print(f'程式運行時間{round(end_time - start_time, 2)}秒')