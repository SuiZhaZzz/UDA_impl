from PIL import Image
import matplotlib.pyplot as plt
import mmseg.models.uda.clsnet.imutils as imtools
import numpy as np

cam = "/root/autodl-tmp/DAFormer/demo/"

img = "/root/autodl-tmp/DAFormer/data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png"
img1 = Image.open(img).convert('RGB')
img1 = imtools.ResizeLong(img1, 256, 512)
img1 = np.array(img1)
img1 = imtools.Crop(img1, 224)
img1 = Image.fromarray(np.uint8(img1))

for i in range(19):
    img2 = Image.open(cam+str(i)+"_cam.png")

    img1 = img1.convert("RGBA")
    img2 = img2.convert("RGBA")

    out = Image.blend(img1, img2, 0.8)
    out.save(cam + "blend_cam_" + str(i) + ".png")
    print(str(i) + " saved" )