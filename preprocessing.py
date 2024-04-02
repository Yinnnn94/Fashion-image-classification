import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.models import load_model
import cv2

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
        for file in (filter_data[filter_data[filter_col] == key]['id'][:100]):
            if str(file) + '.jpg' not in os.listdir(path):
                print(f'{file}不存在')
            else:
                jpg_path = f'{path}\{str(file)}.jpg'
                images = cv2.imread(jpg_path)
                if images.shape != (80, 60, 3):
                    pass
                else:
                    print(images.shape)
                    img_array.append(images.astype('float32')/255)
                    label_zero = np.zeros(len(filter_label))
                    label_zero[i] = 1
                    img_label_counts.append(label_zero)
    return img_array, img_label_counts , filter_label

gender_array, gender_label_counts, gender_label = img_to_array(image_path, 'gender', data)

print(gender_array, gender_label_counts)

# 將label的資料計數
# def label_count_plot(label):
#     value = list(set(label))
#     value_counts = [label.count(v) for v in value]
#     print(value_counts)
#     plt.bar(value, value_counts)
#     plt.show()

# label_count_plot(gender_label_counts)

# 繪製圖片
# def plot_img(array):
#     for i in array:
#         plt.imshow(i, cmap = 'gray')
#         plt.show()
# plot_img(gender_array[:5])
x_train, x_test, y_train, y_test = train_test_split(gender_array, gender_label_counts)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(80, 60, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense((5), activation = 'softmax'))
model.summary()
model.compile(optimizer ='adam', loss ='categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 10, batch_size = 5)
model.save('gender.h5')

gender_pred = model.predict(x_test)
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

fig, ax = plt.subplots(1, 2, figsize = (8, 3))
ax[0].plot(history.history['accuracy'], label='accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy')
ax[1].plot(history.history['loss'], label='Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_title('Loss')
plt.show()

precision = precision_score(np.argmax(y_test, axis=-1), np.argmax(gender_pred, axis = -1), average = 'micro')    
recall = recall_score(np.argmax(y_test, axis=-1), np.argmax(gender_pred, axis = -1), average = 'micro')
F1 = f1_score(np.argmax(y_test, axis=-1), np.argmax(gender_pred, axis = -1), average = 'micro')
CM = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(gender_pred, axis = -1))
print('precision score:',round(precision,4))
print('recall score:',round(recall,4))    
print('f1 score:',round(F1,4))    
plt.title("Confusion Matrix")    
sns.heatmap(CM,annot = True,xticklabels = gender_label, yticklabels = gender_label,cmap = plt.cm.Blues)
plt.show()
end_time = time.time()
print(f'程式運行時間{round(end_time - start_time, 2)}秒')