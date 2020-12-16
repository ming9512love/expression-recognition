import os
from PIL import Image

input_dir=r"C:\local\dateaset\7class\surprise\\"
#例如 input_dir=r"E:\Github\input\\"
output_dir=r"C:\local\dateaset\7class_128\surprise\\"
#例如 output_dir=r"E:\Github\output\\"

filename = os.listdir(input_dir)

size_m = 128
size_n = 128
#这里修改图片尺寸
 
for img in filename:
    image = Image.open(input_dir + img)
    image_size = image.resize((size_m, size_n),Image.ANTIALIAS)
    image_size.save(output_dir+ img)

print("完成！！！")