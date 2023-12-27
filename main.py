import cv2

import torch
from torch import nn
import torchvision
import torchvision.transforms as T

from ultralytics import YOLO

model = YOLO("yolov8m.pt")

vid = cv2.VideoCapture(1)

while(True):
    ret, frame = vid.read()

    results = model.predict([frame[:,0:1280,:],frame[:,1280:2560,:]],save=False)

    for i in range(len(results[0].boxes.xyxy)):
        x1 = int(results[0].boxes.xyxy[i][0])
        x2 = int(results[0].boxes.xyxy[i][2])
        y1 = int(results[0].boxes.xyxy[i][1])
        y2 = int(results[0].boxes.xyxy[i][3])
        cv2.rectangle(frame[:,0:1280,:],(x1,y1),(x2,y2),(255,0,0),5)

    cv2.imshow('frame',frame[:,0:1280,:])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()
