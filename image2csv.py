import os
import csv
import numpy as np
from PIL import Image

'''

这个方法是将Image图片转化为CSV文件
各方法含义在方法体上面都有注释

'''

dataset_path = r'C:\local\dateaset\7class_128_gray2\val.csv'
init_image_path = r'C:\local\dateaset\7class_128_gray2\val'


# 将所有图片转化为csv文件
def writeImage2csv():
    with open(dataset_path, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["emotion", "pixels"])
        all_file_paths = getAllFiles()
        for path in all_file_paths:
            (filepath, tempfilename) = os.path.split(path)
            (shotname, extension) = os.path.splitext(tempfilename)
            emotion = shotname.split("_")[0]
            #usage = shotname.split("_")[1]
            pixels = getSingleFilePixels(path)
            writer.writerow([emotion, pixels])

        csvfile.close()


# 获取单个文件的像素
def getSingleFilePixels(image_path):
    pixels = ""
    image = Image.open(image_path)
    matrix = np.asarray(image)
    for row in matrix:
        for pixel in row:
            pixels+=str(pixel)+" "
    return pixels


# 获取所有的文件名称
def getAllFiles():
    all_file_paths = []
    for root, dirs, files in os.walk(init_image_path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                all_file_paths.append(root + "\\" + file)
    return all_file_paths


writeImage2csv()