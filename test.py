import os
import os.path

f=open('C:/local/dateaset/new/val.csv',encoding="gbk")
content = f.read()

f.close()

a = content.replace("anger","0")
#a = content.replace("disgust","1")
#a = content.replace("fear","2")
#a = content.replace("happiness","3")
#a = content.replace("sadness","4")
#a = content.replace("surprise","5")
#a = content.replace("neutral","6")
#a = content.replace("unknown","7")

with open("C:/local/dateaset/new/val.csv","w",encoding='gbk') as f1:
    f1.write(a)




