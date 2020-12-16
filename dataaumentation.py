import os
import numpy as np
import cv2
from PIL import Image

data_path = 'C:\\local\\dateaset\\manga109_train\\train_backup\\'
aug_path = 'C:\\local\\dateaset\\manga109_train\\train_augmentation2\\'
files = os.listdir(data_path)
a=426
#for file in files:
    #img = Image.open(data_path+file)
    
    #filename1 = os.path.split(file)
    #filename2 = os.path.splitext(filename1[1])
    #b=filename2[0]
    #c=b.split("0")[0]
    
    #out = img.transpose(Image.FLIP_LEFT_RIGHT)
    #out.save(aug_path+str(c)+str(a)+'.jpg')
    #a=a+1

def gasuss_noise(image, mean=0, var=0.0001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

for file in files:
    img = cv2.imread(data_path+file)
    kk=gasuss_noise(img)
    cv2.imwrite(aug_path+"train"+str(a)+'.jpg',kk)
    a=a+1

