# 工具类使得导入的图片都是同样大小

from PIL import Image

def keep_image_size_open(path,size = (224,224)): # 实现等比缩放
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask = mask.resize(size)
    return mask