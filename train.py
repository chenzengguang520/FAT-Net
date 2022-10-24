from torch import nn,optim
import torch
from torch.utils.data import DataLoader
import math
from data import *
from FAT_Net import *
from torchvision.utils import save_image
from Loss import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
weight_path = 'params/fat_net.pth'
data_path = r"D:\class\深度学习\深度学习源码\FAT-Net\dataset"
save_path = 'train_image'

def copy_file(target_dir):
    # 创建文件夹，如果存在，就不再重复创建
    os.makedirs(name=target_dir, exist_ok=True)

# copy_file(f'{save_path}/{1}')

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path),batch_size=2,shuffle=True)
    net = FAT_Net().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('succssful load weight!')

    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = Loss()

    epoch = 1
    while True:
        for i,(image,segment_image) in enumerate(data_loader):
            image,segment_image = image.to(device),segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image,segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()
            if i % 5 == 0:
                print(f'{epoch}--{i}--train_loss====>>{train_loss.item()}')

            if i % 50 == 0:
                torch.save(net.state_dict(),weight_path)

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]
            copy_file(f'{save_path}/{i}')
            # img = torch.stack([_image,_segment_image,_out_image],dim=0)
            save_image(_image,f'{save_path}/{i}/{i}_image.png')
            save_image(_segment_image, f'{save_path}/{i}/{i}_segment_image.png')
            save_image(_out_image, f'{save_path}/{i}/{i}_out_image.png')
            # total += 1
            # for p in opt.param_groups:
            #     p['lr'] = p['lr'] * ((1 - epoch / (total + 1)) ** 0.9)

        epoch += 1