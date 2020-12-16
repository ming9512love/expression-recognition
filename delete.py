import os
#批量删除
path = "C:/local/dateaset/manga109_cut2/"
files = os.listdir(path)
for i ,f in enumerate(files):
    if f.find("0")>=0 or f.find("1")>=0 or f.find("2")>=0 or f.find("3")>=0 or f.find("4")>=0:
        print(i)
        os.remove(path+f)
