## Fashion image classification👚
🎯Get multiple tags for images without manually adding tags to images.  
Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small  
Total number of photos: 44.4k  
### Label Flow
+ It has 4 main tags (master category, subcategory, gender, and usage)
+ First, predict which main category the picture belongs to. Different types of main categories need to be predicted in different categories.
+ If it is predicted that it is clothing, then you need to identify the gender, clothing type, and usage occasion; if it is an accessory, you need to identify which gender the accessory belongs to.
![image](https://github.com/Yinnnn94/Fashion-image-classification/blob/main/Readme_img/LabelFlow.drawio.png)
