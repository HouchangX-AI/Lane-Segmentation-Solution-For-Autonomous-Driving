import config as cfg
import pandas as pd

from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn

from utils.dataset import BasicDataset
from models.unet_base import unet_base
from eval import eval_net
from tqdm import tqdm
from dice_loss_1 import MulticlassDiceLoss

from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from torchvision import transforms

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main():
    # network = 'deeplabv3p'
    # save_model_path = "./model_weights/" + network + "_"
    # model_path = "./model_weights/" + network + "_0_6000"
    data_dir = '/env/data_list_for_Lane_Segmentation/train.csv'
    val_percent = .1
    
    epochs = 9

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    training_dataset = LaneDataset("/env/data_list_for_Lane_Segmentation/train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                              ScaleAug(), CutOut(32, 0.5), ToTensor()]))

    training_data_batch = DataLoader(training_dataset, batch_size=2,
                                     shuffle=True, drop_last=True, **kwargs)




    dataset = BasicDataset(data_dir, img_size=cfg.IMG_SIZE, crop_offset = cfg.crop_offset)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = unet_base(cfg.num_classes, cfg.IMG_SIZE)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, betas=(0.9, 0.99))

    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = MulticlassDiceLoss()

    model.train()
    epoch_loss = 0

    dataprocess = tqdm(training_data_batch)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(), mask.cuda()
            image = image.to(torch.float32).requires_grad_()
            mask = mask.to(torch.float32).requires_grad_()

            masks_pred = model(image)
            masks_pred = torch.argmax(masks_pred, dim=1)
            masks_pred = masks_pred.to(torch.float32)
            mask = mask.to(torch.float32)

            # print('mask_pred:', masks_pred)
            # print('mask:', mask)
            loss = bce_criterion(masks_pred, mask) + dice_criterion(masks_pred, mask)
            epoch_loss += loss.item()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # for epoch in range(epochs):
    #     model.train()
    #
    #     epoch_loss = 0
    #     with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
    #         for batch in train_loader:
    #             imgs = batch['image']
    #             true_masks = batch['mask']
    #
    #             imgs = imgs.to(device=device, dtype=torch.float32)
    #             mask_type = torch.float32 if cfg.num_classes == 1 else torch.long
    #             true_masks = true_masks.to(device=device, dtype=mask_type)
    #
    #             masks_pred = model(imgs)
    #
    #             loss = bce_criterion(masks_pred, true_masks) + dice_criterion(masks_pred, true_masks)
    #             epoch_loss += loss.item()
    #
    #             pbar.set_postfix(**{'loss (batch)': loss.item()})
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #             pbar.update(imgs.shape[0])
    #
    #             global_step += 1
    #             if global_step % (len(dataset) // (10 * cfg.batch_size)) == 0:
    #                 val_score = eval_net(model, val_loader, device, n_val)
    #                 print ('val_score:', val_score)
                    # if cfg.num_classes > 1:
                    #     logging.info('Validation cross entropy: {}'.format(val_score))
                    #     writer.add_scalar('Loss/test', val_score, global_step)

                    # else:
                    #     logging.info('Validation Dice Coeff: {}'.format(val_score))
                    #     writer.add_scalar('Dice/test', val_score, global_step)

                    # writer.add_images('images', imgs, global_step)
                    # if net.n_classes == 1:
                    #     writer.add_images('masks/true', true_masks, global_step)
                    #     writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)


main()