
import os
import pandas as pd
from sklearn.utils import shuffle


label_list = []
image_list = []

image_dir = '/env/images_for_BAIDU_Lane_Segmentation/Image_Data/'
label_dir = '/env/images_for_BAIDU_Lane_Segmentation/Gray_Label/'
save_path = '/env/data_list_for_Lane_Segmentation/'


for road in os.listdir(image_dir):
    for record in os.listdir(os.path.join(image_dir, road)):
        for camera in os.listdir(os.path.join(image_dir, road, record)):
            for image_name in os.listdir(os.path.join(image_dir, road, record, camera)):
                label_image_name = image_name.replace('.jpg', '_bin.png')
                image_path = os.path.join(image_dir, road, record, camera, image_name)
                label_path = os.path.join(image_dir, 'Label_' + str.lower(road), 'Label', record, camera, label_image_name)
                image_list.append(image_path)
                label_list.append(label_list)
print (len(image_list))
print (len(label_list))
csv_file = pd.DataFrame({'image' : image_list, 'label' : label_list})
csv_file = shuffle(csv_file)
csv_file.to_csv(os.path.join(save_path, 'train.csv'), index=False)


