import os

path = 'C:\\local\\dateaset\\7class_128_gray2\\anger'

filelist = os.listdir(path)
#total_num = len(filelist)
i = 0
name = 'anger'
for item in filelist:
    src = os.path.join(path, item)
    # dst = os.path.join(path,'val'+str(i)+'.jpg')
    if len(str(i))==1:
        dst = os.path.join(path,str(name)+'_00'+str(i)+'.jpg')
    elif len(str(i))==2:
        dst = os.path.join(path,str(name)+'_0'+str(i)+'.jpg')
    else:
        dst = os.path.join(path,str(name)+'_'+str(i)+'.jpg')
    os.rename(src,dst)
    i=i+1