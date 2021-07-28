from PIL import Image
from PIL import ImageFilter
import os
import torch
import torchvision
# im=Image.open('test.jpg')
# out=im.resize((20,20))#
# out.save("out.png","PNG") #
# loader使用torchvision中自带的transforms函数
print(111111111111)
print(222222222222)
loader = torchvision.transforms.Compose \
([torchvision.transforms.ToTensor()])
unloader = torchvision.transforms.ToPILImage()
def file_name(file_dir):
    print(333333333333)
    for root, dirs, files in os.walk(file_dir):
        count = 1
        print(files)
        # 当前文件夹所有文件
        for i in files:
            # 判断是否以.jpg结尾
            if i.endswith('.png'):
                # 如果是就改变图片像素为28 28
                i =  file_dir + i
                im = Image.open(i)
                # #add Gussian blur
                # im = im.filter(ImageFilter. \
                #         GaussianBlur(radius = 3.5))
                # # PIL picture trans floatTensor
                # im = loader(im).unsqueeze(0)
                # im = im.float()
                # #add Gussian noise
                # add_noise = \
                #     torch.FloatTensor(im.size()).normal_(mean=0, std=25 / 255.)
                # im = im + add_noise
                # # image trans PIL
                # im = im.cpu().clone()
                # im = im.squeeze(0)
                # im = unloader(im)

                # #resize image
                # im = im.resize((224, 224))

                im.save('./HQ_GT/' + str(count) + '.png', 'PNG')
                count += 1
                print(i)


file_name('/mnt/lustre/share/wanghaoran/data2/FFHQ/images1024x1024/69000/')  # 当前文件夹