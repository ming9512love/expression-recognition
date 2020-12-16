from PIL import Image

import os

file_dir = 'C:\\local\\dateaset\\7class_128_rbg\\surprise'

out_dir = 'C:\\local\\dateaset\\7class_128_gray\\surprise'

img = os.listdir(file_dir)

for i in img:

    print(i)
    I = Image.open(file_dir+"/"+i)
    L = I.convert('L')
    L.save(out_dir+"/"+i)

