from preprocessing import Fashion_image
import time
import pandas as pd

start_time = time.time()
data = pd.read_excel(r"after_cleaning.xlsx", sheet_name='刪除後data', usecols=['id', 'gender', 'masterCategory', 'subCategory', 'season', 'usage'])
data = data.dropna()

img_path = 'images'

FIC = Fashion_image(img_path, data, 'season')
FIC.split_data()
# FIC.plot_head_5_image()
# img_array, img_label_array, label = FIC.img_to_array()
# x_train, x_test, y_train, y_test = FIC.split_data(img_array, img_label_array)
# history, model = FIC.model_trainning(x_train, x_test, y_train, y_test, label)
# FIC.evaluate_model(model, x_test, y_test, history, label)

end_time = time.time()
print(f'程式運行時間{round(end_time - start_time, 2)}秒')