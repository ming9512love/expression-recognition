import os

path = 'C:\\local\\dateaset\\new\\val'

filelist = os.listdir(path)
total_num = len(filelist)
i = 0
name = 'surprise'
for item in filelist:
    src = os.path.join(path, item)
    dst = os.path.join(path,'val'+str(i)+'.jpg')

    os.rename(src,dst)
    i=i+1