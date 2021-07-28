from PIL import ImageFilter
from PIL import Image
import torch
import torchvision

def addBlur_and_addNoise(image, blurLeve, noiseLeve):
    """

    @param image: FloatTensor
    @param blurLeve:
    @param noiseLeve:
    """
    # loader使用torchvision中自带的transforms函数
    loader = torchvision.transforms.Compose \
        ([torchvision.transforms.ToTensor()])
    unloader = torchvision.transforms.ToPILImage()

    # add Gussian noise
    add_noise = \
        torch.FloatTensor(image.size()).normal_(mean=0, std=noiseLeve / 255.)
    image = image + add_noise

    #image trans PIL
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)

    #add Gussian blur
    image = image.filter(ImageFilter. \
                        GaussianBlur(radius=blurLeve))

    # PIL picture trans floatTensor
    image = loader(image).unsqueeze(0)
    image = image.float()

    return image





    # # PIL read image and add Gussian blur
    # img = Image.open(image)
    # img_02 = img.filter(ImageFilter. \
    #                     GaussianBlur(radius=blurLeve))
    #
    # # PIL picture trans floatTensor
    # image = loader(img_02).unsqueeze(0)
    # image = image.float()
    #
    # #add Gussian noise
    # add_noise = \
    # torch.FloatTensor(image_data.size()).normal_(mean=0, std=noiseLeve / 255.)
    # image = image + add_noise


# # loader使用torchvision中自带的transforms函数
# loader = torchvision.transforms.Compose\
#     ([torchvision.transforms.ToTensor()])
# unloader = torchvision.transforms.ToPILImage()
#
# #PIL read image and add Gussian blur
# img = Image.open("img.png")
# img_02 = img.filter(ImageFilter.\
#                     GaussianBlur(radius=0.0))
# # print(type(img_02))
# img_02.save("img_blur.jpg")
# #PIL picture trans floatTensor
# image = loader(img_02).unsqueeze(0)
# image = image.float()
# # print(image.type())
# # print(image.shape)
#
# #add Gussian noise
# add_noise = \
# torch.FloatTensor(image.size()).normal_(mean=0, std=50 / 255.)
# image = image + add_noise
#
# image = image.cpu().clone()
# image = image.squeeze(0)
# image = unloader(image)
#
# image.save("img_blur_noise1.jpg")


