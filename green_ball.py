import math
import cv2
import numpy as np

from collections import deque
import imutils

vid = cv2.VideoCapture(1)

while(True):
    #frame: 1280x960
    ret, frame = vid.read()
    frame = frame[:,0:1280,:]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    gmin = np.array([25,52,72])
    gmax = np.array([102,255,255])
    pts = deque()
    
    g_clamp = cv2.inRange(hsv,gmin,gmax)
    g_clamp = cv2.erode(g_clamp, None, iterations=2)
    g_clamp = cv2.dilate(g_clamp, None, iterations=2)

    cnts = cv2.findContours(g_clamp.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = (-1,-1)
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
	    
    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()
