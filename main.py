import math
import cv2

import torch
from torch import nn
import torchvision
import torchvision.transforms as T

from ultralytics import YOLO

model = YOLO("yolov5su.pt")

vid = cv2.VideoCapture(1)

pixelAngle = 45 / math.sqrt(1280**2 + 960**2)
d = 5

def getDistanceSquare2D(coord0, coord1):
    x0 = (coord0[0] + coord0[1]) / 2
    y0 = (coord0[2] + coord0[3]) / 2
    x1 = (coord1[0] + coord1[1]) / 2
    y1 = (coord1[2] + coord1[3]) / 2
    return (x1 - x0) ** 2 + (y1 - y0) ** 2

def getDistance3D(coord0, coord1):
    #calculate x and y with the center being (0,0)
    x0 = (coord0[0] + coord0[1]) / 2 - 640
    y0 = (coord0[2] + coord0[3]) / 2 - 480
    x1 = (coord1[0] + coord1[1]) / 2 - 640
    y1 = (coord1[2] + coord1[3]) / 2 - 480
    #calculate their angle to the camera plane
    a = math.sqrt(x0**2 + y0**2) * pixelAngle
    b = math.sqrt(x1**2 + y1**2) * pixelAngle
    a = abs(90 - a)
    b = abs(90 - b)
    #put the object into 3 sections: section 0 is on the left of the left cam
    #section 1 is between the left and right cams, section 2 is on the right
    #of the right cam
    section = 0
    if x0 * x1 < 0:
        section = 1
    else:
        if a > b:
            section = 2

    #calculate the distance by section
    if section == 0:
        x = d * math.cos(math.radians(b)) / (math.cos(math.radians(a)) - math.cos(math.radians(b)))
        y = x * math.tan(math.radians(b))
        return math.sqrt((x+d/2)**2+y**2)
    elif section == 2:
        x = d * math.cos(math.radians(a)) / (math.cos(math.radians(b)) - math.cos(math.radians(a)))
        y = x * math.tan(math.radians(a))
        return math.sqrt((x+d/2)**2+y**2)
    else:
        x = d * math.cos(math.radians(a)) / (math.cos(math.radians(a)) + math.cos(math.radians(b)))
        y = x * math.tan(math.radians(a))
        return math.sqrt((d/2-x)**2+y**2)

while(True):
    #frame: 1280x960
    ret, frame = vid.read()

    results = model.predict([frame[:,0:1280,:],frame[:,1280:2560,:]],save=False)

    objs = []
    depth = []

    for i in range(len(results[0].boxes.cls)):
        index = -1
        for j in range(len(results[1].boxes.cls)):
            if results[0].boxes.cls[i] == results[1].boxes.cls[j]:
                if index == -1:
                    index = j
                else:
                    if getDistanceSquare2D(results[0].boxes.xyxy[i],results[1].boxes.xyxy[j]) < getDistanceSquare2D(results[0].boxes.xyxy[i],results[1].boxes.xyxy[index]):
                        index = j
        if index != -1:
            objs.append([i,index])
            depth.append(getDistance3D(results[1].boxes.xyxy[index],results[0].boxes.xyxy[i]))

    for i in range(len(objs)):
        x1 = int(results[0].boxes.xyxy[objs[i][0]][0])
        x2 = int(results[0].boxes.xyxy[objs[i][0]][2])
        y1 = int(results[0].boxes.xyxy[objs[i][0]][1])
        y2 = int(results[0].boxes.xyxy[objs[i][0]][3])
        cv2.rectangle(frame[:,0:1280,:],(x1,y1),(x2,y2),(255,0,0),5)
        cv2.putText(frame[:,0:1280,:],str(depth[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)

    cv2.imshow('frame',frame[:,0:1280,:])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()
