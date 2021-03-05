#contains all the functions related to identifying park/vegetation/green cover in the images
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


def get_min_park_distance(contours,midpoint):
    #function to find the minimum among the distances between the midpoint and all of the park contours
    park_distances = []
    for i in contours:
        distance = abs(int(cv2.pointPolygonTest((i), midpoint, True)))
        park_distances.append(distance) # all distances from midpoint to all parks in an image
    return min(park_distances)

def get_park_contours(image,midpoint):
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_frame, (40, 25, 25), (70, 255, 255))  # to get only the green areas on map
    imask = mask > 0
    green = np.zeros_like(image, np.uint8)
    green[imask] = image[imask]
    onlypark_img = green.copy()
    #cv2.imshow("parks", onlypark_img)
    cv2.waitKey(0)
    if onlypark_img.any() > 0: #park exists in the image
        park_contours = findContr(edgedetect(onlypark_img))
     #   print('len(park_contours)', len(park_contours))
        min_park_dist = get_min_park_distance(park_contours, midpoint)
    else:#no park in the given image
        min_park_dist = -1 # there are cases where green cover doesnt exist in an image,hence a default value given

    return min_park_dist


#image = cv2.imread('Images_all/tmp_8.png')
#cv2.imshow("Original image", image)
#cv2.waitKey(0)
#min_park_dist = get_park_contours(image,midpoint(image))
#print(min_park_dist)