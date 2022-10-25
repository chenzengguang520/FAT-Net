import os

import torch
from torch.utils.data import DataLoader
from FAT_Net import *
from utils import keep_image_size_open
from data import *
from torchvision.utils import save_image

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    # transforms.ToTensor(),

])

net = FAT_Net().cuda()
save_path = 'test_image'
def copy_file(target_dir):
    # 创建文件夹，如果存在，就不再重复创建
    os.makedirs(name=target_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = 'params/fat_net.pth'
data_path = r"D:\class\深度学习\深度学习源码\FAT-Net\dataset"
if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=2, shuffle=True)
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successful')

    else :
        print('no loading')

    # _input = input('please input image path:')

    for i, (image, segment_image) in enumerate(data_loader):
        image, segment_image = image.to(device), segment_image.to(device)
        # img = keep_image_size_open(image)
        # img_data = transform1(img).cuda()
        # img_data = torch.unsqueeze(img_data,dim=0)
        # print(img_data.shape)
        # print(image.shape)
        out = net(image)
        # save_image(out,'result/result.jpg')
        # print(out)
        _image = image[0]
        _image = transform(_image)
        print(_image.shape)
        # _image = transform(_image)
        _segment_image = segment_image[0]
        _out_image = out[0]
        # copy_file(f'{save_path}/{i}')
        img = torch.stack([_image,_out_image,_segment_image],dim=0)
        # print(i)
        save_image(img, f'{save_path}/{i}.png')