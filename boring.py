import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional
import albumentations as A
import segmentation_models_pytorch as smp

# データローダーの作成
additional_image_targets = {f"image{i+1}": "image" for i in range(8 - 1)}
additional_label_targets = {f"mask{i+1}": "mask" for i in range(8 - 1)}
additional_targets8 = dict(additional_image_targets, **additional_label_targets)

data_transforms8 = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
      A.OneOf([
          A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
          A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
      ], p=0.25),],
      additional_targets = additional_targets8, p=1.0)

class CustomDataset(torch.utils.data.Dataset):
  def __init__(
      self,
      df
      ):
    self.depth = 8
    self.imgpath_list = [[df.iloc[i,:][f"imgpath_{h}"] for h in range(self.depth)] for i in range(len(df))]
    self.labelpath_list = [[df.iloc[i,:][f"labelpath_{h}"] for h in range(self.depth)] for i in range(len(df))]
    self.transform8 = data_transforms8

  def __getitem__(self, i):
    imgpathes = self.imgpath_list[i]
    labelpathes = self.labelpath_list[i]
    for j in range(self.depth):  #画像とラベルデータの読み込み
      img = cv2.imread(imgpathes[j])
      img = cv2.resize(img, dsize = (256, 256))
      label = Image.open(labelpathes[j])
      label = np.asarray(label)
      label = cv2.resize(label, dsize = (256, 256))
      if j == 0:  #深さ方向に結合
        img_3D = [img]
        label_3D = [label]
      else:
        img_3D = np.vstack([img_3D, [img]])
        label_3D = np.vstack([label_3D, [label]])
      
    d1 = {"image": img_3D[0,:,:,:]}  #Albumentationsに代入する為の辞書型データを作成
    d2 = {f"image{i+1}": img_3D[i+1,:,:,:] for i in range(self.depth - 1)}
    d3 = {"mask": label_3D[0,:,:]}
    d4 = {f"mask{i+1}": label_3D[i+1,:,:] for i in range(self.depth - 1)}
    dic = dict(d1, **d2, **d3, **d4)

    transformed = self.transform8(**dic)

    for j in range(self.depth):
      if j == 0:  #データ拡張後のデータを再度深さ方向に結合
        img_3D = [transformed["image"]]
        label_3D = [transformed["mask"]]
      else:
        img_3D = np.vstack([img_3D, [transformed[f"image{j}"]]])
        label_3D = np.vstack([label_3D, [transformed[f"mask{j}"]]])

    img_3D = img_3D/255  #RGBの値を0~1に
    img_3D = torch.from_numpy(img_3D.astype(np.float32)).clone()
    img_3D = img_3D.permute(3, 0, 1, 2)  #（チャンネル数、深さ、高さ、幅）に変換
    label_3D = torch.from_numpy(label_3D.astype(np.float32)).clone()
    label_3D = torch.nn.functional.one_hot(label_3D.long(), num_classes=4)
    label_3D = label_3D.to(torch.float32)
    label_3D = label_3D.permute(3, 0, 1, 2)  #（チャンネル数、深さ、高さ、幅）に変換
    data = {"img": img_3D, "label": label_3D}
    return data
  
  def __len__(self):
    return len(self.imgpath_list)


class valtest_Dataset(torch.utils.data.Dataset):  #Albumentationsによるデータ拡張を行わない
  def __init__(
      self,
      df
      ):
    self.depth = 8
    self.imgpath_list = [[df.iloc[i,:][f"imgpath_{h}"] for h in range(self.depth)] for i in range(len(df))]
    self.labelpath_list = [[df.iloc[i,:][f"labelpath_{h}"] for h in range(self.depth)] for i in range(len(df))]

  def __getitem__(self, i):
    imgpathes = self.imgpath_list[i]
    labelpathes = self.labelpath_list[i]
    for j in range(self.depth):
      img = cv2.imread(imgpathes[j])
      label = Image.open(labelpathes[j])
      label = np.asarray(label)
      img = cv2.resize(img, dsize = (256, 256))
      label = cv2.resize(label, dsize = (256, 256))
      if j == 0:
        img_3D = [img]
        label_3D = [label]
      else:
        img_3D = np.vstack([img_3D, [img]])
        label_3D = np.vstack([label_3D, [label]])

    img_3D = img_3D/255
    img_3D = torch.from_numpy(img_3D.astype(np.float32)).clone()
    img_3D = img_3D.permute(3, 0, 1, 2)
    label_3D = torch.from_numpy(label_3D.astype(np.float32)).clone()
    label_3D = torch.nn.functional.one_hot(label_3D.long(), num_classes=4)
    label_3D = label_3D.to(torch.float32)
    label_3D = label_3D.permute(3, 0, 1, 2)
    data = {"img": img_3D, "label": label_3D}
    return data
  
  def __len__(self):
    return len(self.imgpath_list)