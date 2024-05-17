import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import time
import seaborn as sns
import shutil
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.models import load_model
from keras.callbacks import TensorBoard, EarlyStopping
import datetime 


class Fashion_image():
    
    def __init__(self, path, data, filter_col):
        self.path = path
        self.data = data
        self.filter_col = filter_col

    def plot_head_5_image(self):
        for category in self.data[self.filter_col].unique():
            fig, axs = plt.subplots(1, 5, figsize=(15, 10))
            for i, ax in zip(self.data[self.data[self.filter_col] == category]['id'][:5], axs.ravel()):
                jpg_path = f'{self.path}/{str(i)}.jpg'
                images = plt.imread(jpg_path)
                ax.imshow(images, cmap='gray')
                ax.set_title(f'{category}:{i}')
        plt.show()

    def split_data(self):
        train_dir = f'{self.path}/to/train'
        test_dir = f'{self.path}/to/test'
        all_files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]

        train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for file in train_files:
            shutil.move(os.path.join(self.path, file), os.path.join(train_dir, file))

        # 移動文件到測試集資料夾
        for file in test_files:
            shutil.move(os.path.join(self.path, file), os.path.join(test_dir, file))

        print("文件分割完成！")

    def img_to_array(self):
        img_list = []
        img_label_counts = []
        filter_data = self.data[['id', self.filter_col]]
        filter_name = filter_data[self.filter_col].unique()
        filter_label = {name: i for i, name in enumerate(filter_name)}
        for i, key in enumerate(filter_label):
            print(i, key)
            for file in (filter_data[filter_data[self.filter_col] == key]['id']):
                if str(file) + '.jpg' not in os.listdir(self.path):
                    print(f'{file}不存在')
                else:
                    jpg_path = f'{self.path}/{str(file)}.jpg'
                    images = plt.imread(jpg_path)
                    if images.shape != (80, 60, 3):
                        pass
                    else:
                        img_list.append(images.astype('float32') / 255)
                        label_zero = np.zeros(len(filter_label))
                        label_zero[i] = 1
                        img_label_counts.append(label_zero)
        img_array = np.array(img_list)
        img_label_array = np.array(img_label_counts)
        return img_array, img_label_array, filter_label




    def model_trainning(self, x_train, x_test, y_train, y_test, filter_label):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(80, 60, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense((y_train.shape[1]), activation='softmax'))
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience = 5)
        history = model.fit(x_train, y_train, epochs=50, batch_size = 32, callbacks=[tensorboard_callback, early_stopping_callback])

        model.save(f'{self.filter_col}.h5')
        return history, model

    def evaluate_model(self, model, x_test, y_test, history, label):
        pred = model.predict(x_test)
        score = model.evaluate(x_test, y_test)

        # Plots with the history
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].plot(history.history['accuracy'], label='accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_title('Accuracy')
        ax[1].plot(history.history['loss'], label='Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Loss')
        plt.show()
        precision = precision_score(np.argmax(y_test, axis=-1), np.argmax(pred, axis=-1), average='micro')
        recall = recall_score(np.argmax(y_test, axis=-1), np.argmax(pred, axis=-1), average='micro')
        F1 = f1_score(np.argmax(y_test, axis=-1), np.argmax(pred, axis=-1), average='micro')
        CM = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(pred, axis=-1))

        # Some metrics
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print('precision score:', round(precision, 4))
        print('recall score:', round(recall, 4))
        print('f1 score:', round(F1, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=label.keys())
        disp.plot(cmap='Blues')
        plt.show()


