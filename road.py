#contains all the functions related to identifying roads in the images

import cv2
import numpy as np


def midpoint(image):
    a = image.shape
    y = int((a[0])/2)
    x = int((a[1])/2)
    return (x,y)

def findContr(image):
    contours, _  = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return contours

def edgedetect(image):
    # image is converted to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edged = cv2.Laplacian(src=gray, ddepth=cv2.CV_8U, ksize=5)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged

def get_min_road_distance(contours,midpoint):
    road_distance = []
    for i in contours:

        distance = abs(int(cv2.pointPolygonTest((i), midpoint, True)))
        road_distance.append(distance) # all distances from midpoint to all roads
    return min(road_distance)

def get_road_contours(image,midpoint):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 150)
   # cv2.imshow("Roads", canny)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    road_contours = findContr(canny)
    min_road_dist = get_min_road_distance(road_contours,midpoint)
    return min_road_dist

#image = cv2.imread('Images_all/tmp_2.png')
#min_road_dist = get_road_contours(image,midpoint(image))