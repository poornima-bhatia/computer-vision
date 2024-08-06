import numpy as np
import cv2
from PIL import Image

def get_limits(color):
    c = np.uint8([[color]])
    hsv_c = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    
    lowerLimit = hsv_c[0][0][0] - 10 , 100 , 100
    upperLimit = hsv_c[0][0][0] + 10 , 255 , 255
    
    lowerLimit = np.array(lowerLimit , dtype=np.uint8)
    upperLimit = np.array(upperLimit , dtype=np.uint8)
    
    return lowerLimit , upperLimit
yellow = [0, 255, 255]
capture = cv2.VideoCapture(0)
while True:
    ret , frame = capture.read()
    hsvImg = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
    lowerLimit , upperLimit = get_limits(color=yellow)
    mask = cv2.inRange(hsvImg ,lowerLimit , upperLimit)
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()
    print(bbox)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame , (x1, y1) , (x2, y2) , (0, 255, 0) , 5)
    cv2.imshow('frame' , frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()