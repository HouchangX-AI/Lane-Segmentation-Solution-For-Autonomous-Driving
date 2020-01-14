from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor


kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
training_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                          ScaleAug(), CutOut(32,0.5), ToTensor()]))


training_data_batch = DataLoader(training_dataset, batch_size=16,
                                 shuffle=True, drop_last=True, **kwargs)

dataprocess = tqdm(training_data_batch)
for batch_item in dataprocess:
    image, mask = batch_item['image'], batch_item['mask']
    if torch.cuda.is_available():
        image, mask = image.cuda(), mask.cuda()
    # this is aimed that debug your new method
    # print(image.size())
    # print(mask.size())
    # image = image.numpy()
    # print(type(image))
    # plt.imshow(np.transpose(image[0],(1,2,0)))
    # plt.show()
    # plt.imshow(mask[0])
    #
    # plt.show()




