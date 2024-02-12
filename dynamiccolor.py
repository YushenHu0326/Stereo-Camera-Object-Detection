import math
import cv2

import torch
from torch import nn
import torchvision
import torchvision.transforms as T

import numpy as np

from ultralytics import YOLO

def clamp(t, minimum, maximum):
    if t < minimum:
        return minimum
    elif t > maximum:
        return maximum
    else:
        return t

def v3interp(t, a, b):
    return [a[0]*t+b[0]*(1-t),a[1]*t+b[1]*(1-t),a[2]*t+b[2]*(1-t)]

def colorTransform(src, x1, x2, y1, y2, points, inColor, outColor):
    for x in range(x1, x2):
        for y in range(y1, y2):
            if cv2.pointPolygonTest(points, (x,y), False) > 0:
                error = abs(src[y][x][0] - inColor[0]) + abs(src[y][x][1] - inColor[1]) + abs(src[y][x][2] - inColor[2])
                src[y][x] = [error, error, error]
                #src[y][x] = v3interp(clamp(500-error,0,500)/500, outColor, src[y][x])

model = YOLO("yolov5n-Seg.pt")
names = model.names

vid = cv2.VideoCapture(1)

while(True):
    #frame: 1280x960
    ret, frame = vid.read()
    img = cv2.resize(frame[:,0:1280,:], (640,480))

    results = model.predict([img],save=False)

    objs = []

    for i in range(len(results[0].boxes.cls)):
            objs.append(i)

    for i in range(len(objs)):
        x1 = int(results[0].boxes.xyxy[objs[i]][0])
        x2 = int(results[0].boxes.xyxy[objs[i]][2])
        y1 = int(results[0].boxes.xyxy[objs[i]][1])
        y2 = int(results[0].boxes.xyxy[objs[i]][3])
        points = np.int32([results[0].masks.xy[i]])
        if names[int(results[0].boxes.cls[i])].lower() == "bottle":
            colorTransform(img, x1, x2, y1, y2, points, [255,0,0],[0,255,0])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
        cv2.putText(img,names[int(results[0].boxes.cls[i])],(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)

    cv2.imshow('frame',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()
