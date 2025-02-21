import cv2 as cv
import numpy as np

images = cv.VideoCapture(0)

COLOR_BOUNDS = {
    "yellow": [(10, 210, 57), (38, 255, 255)],
    "red1": [(0, 222, 23), (9, 255, 243)],
    "red2": [(160, 222, 23), (255, 255, 243)],
    "blue": [(110, 240, 0), (130, 255, 225)],
}

while True:
    ret, image = images.read()
    assert ret is not False, 'fehwfuwh'

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_image, (90, 120, 0), (140, 255, 255))

    low_threshold = 60
    high_threshold = 100

    kernel = np.ones((3,3))
    erosion = cv.morphologyEx(mask, kernel = kernel, op = cv.MORPH_OPEN)

    edges = cv.Canny(erosion, low_threshold, high_threshold)
   
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])

    if len(contours) > 0:
        largestContour = max(contours, key=cv.contourArea)
        erect = cv.minAreaRect(largestContour)
        (x,y), (w,h), angle = erect
        box = cv.boxPoints(erect)
        box = np.int32(box)

        if 50 < w < 170 and 50 < h < 170:
            cv.polylines(image, [box], isClosed = True, color = (0, 255, 0), thickness = 2)
            cv.circle(image, (int(x),int(y)), radius = 2, color = (0, 0, 255), thickness = 2)

            if w < h:
                angle = angle + 90
            
            print(angle, w, h)
            cv.putText(image, 'de', org = (50,50), fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1.1, color=(0, 0, 255), thickness = 2)
        
    cv.imshow('fiojew', image)

    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
