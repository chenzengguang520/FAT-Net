from torch.utils.data import Dataset
import os
import torch
from utils import *
from torchvision import transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),

])
transform1 = transforms.Compose([
    transforms.ToTensor(),

])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.namer = os.listdir(os.path.join(path,'ISBI2016_ISIC_Part1_Test_Data')) #获取‘SegmentationClass’目录下面的所有文件
        self.name = os.listdir(os.path.join(path,'ISBI2016_ISIC_Part1_Test_GroundTruth')) #获取‘SegmentationClass’目录下面的所有文件

    def __len__(self):
        return len(self.name)

    def __getitem__(self, item):
        segment_name = self.name[item] # xx.png
        segment_namer = self.namer[item]
        segment_path = os.path.join(self.path,'ISBI2016_ISIC_Part1_Test_GroundTruth',segment_name)
        image_path = os.path.join(self.path,'ISBI2016_ISIC_Part1_Test_Data',segment_namer.replace('png','jpg'))
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform1(image),transform(segment_image)

if __name__ == '__main__':
    data = MyDataset(r"D:\class\深度学习\深度学习源码\FAT-Net\dataset")
    # print(os.listdir(os.path.join(r"D:\class\深度学习\深度学习源码\FAT-Net\dataset", 'ISBI2016_ISIC_Part1_Training_GroundTruth')))
    print(data[0][0].shape)
    print(data[0][1].shape)
