import cv2
import xml.dom.minidom as xmldom
import os
import glob

path = "C:/Users/Zhou/Downloads/Manga109/Manga109_released_2020_09_26/annotations"
files = glob.glob(path + '\\*.xml')

for file in files:
    #分离xml文件名，取消后缀
    filename1 = os.path.split(file)
    filename2 = os.path.splitext(filename1[1])
    ##print(filename2[0])
    xml_file = xmldom.parse(file)
    #根元素
    root = xml_file.documentElement
    #找face标签的
    face=root.getElementsByTagName("face")
    num=len(face)
    j=200
    for i in range(200,num):
        xmin = face[i].getAttribute("xmin")
        xmax = face[i].getAttribute("xmax")
        ymin = face[i].getAttribute("ymin")
        ymax = face[i].getAttribute("ymax")
        a,b=int(xmax)-int(xmin),int(ymax)-int(ymin)
        if j<num:
            if (a>=64 and b>=64):
                #寻找父元素
                parent = face[i].parentNode
                index = parent.getAttribute("index")
                print(int(xmin), int(xmax), int(ymin), int(ymax))
                if len(str(index)) == 1:
                    img = cv2.imread("C:/Users/Zhou/Downloads/Manga109/Manga109_released_2020_09_26/images/"+str(filename2[0])+"/00"+str(index)+'.jpg')
                elif len(str(index)) == 2:
                    img = cv2.imread("C:/Users/Zhou/Downloads/Manga109/Manga109_released_2020_09_26/images/"+str(filename2[0])+"/0"+str(index)+'.jpg')
                else:
                    img = cv2.imread("C:/Users/Zhou/Downloads/Manga109/Manga109_released_2020_09_26/images/"+str(filename2[0])+"/"+str(index)+'.jpg')
                ##print(img.shape)
                cropped = img[int(ymin):int(ymax),int(xmin):int(xmax)]  # 裁剪坐标为[ymin:ymax, xmin:xmax]
                cv2.imwrite("C:/local/dateaset/manga109_cut4/"+str(filename2[0])+str(j)+'.jpg', cropped)
                j=j+1
            else:
                continue
        else:
            continue