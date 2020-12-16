
import os,sys
import random
import shutil
 
 
def copyFile(fileDir):
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir, 159)
    print(sample)
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
 
 
if __name__ == '__main__':
    # open /textiles
    path = "C:\\local\\dateaset\\7class_128_gray2\\surprise"
    dirs = os.listdir(path)
    
    # output all folds
    for file in dirs:
        print(file)
        
        filename = "C:\\local\\dateaset\\新建文件夹\\train"
        
        fileDir = path + "/"
        tarDir = filename + "/"
        copyFile(fileDir)