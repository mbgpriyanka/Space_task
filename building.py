#contains all the functions related to identifying buildings in the images

import cv2
import numpy as np

def midpoint(image):
    a = image.shape
    y = int((a[0])/2)
    x = int((a[1])/2)
    return (x,y)

def findContr(edgeImg):
    _, contours, _  = cv2.findContours(edgeImg.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return contours

def detect_nearestbuilding(image,hb,contours,midpoint): #get the distance to the nearest building from shop location
    #building_contours = get_building(image)
    building_distances=[]
    for i in contours:
        # distance between shop wall and other buildings
        if np.array_equal(hb,i): # if the shop resides in the same building
            wd_to_otherbuilding_distance = 0
        else:
            wd_to_otherbuilding_distance = abs(int(cv2.pointPolygonTest((i),midpoint,True)))
            building_distances.append(wd_to_otherbuilding_distance)

    return min(building_distances)


def check_shop_inside_building(image,midpoint,contours):
    check = -1
    for i in contours:
        check = cv2.pointPolygonTest((i),midpoint,False)
        #check = 0 on the boundary,=1 inside,=-1 outside
        if check == 1:
           home_building = (i)
           break
    return check,home_building


def get_building_contours(img,midpoint):
    h, w = img.shape[:2]

    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (50, 50, 50), 1)

    # convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow('HSV',hsv)
    #cv2.waitKey(0)
    low_yellow = (0, 10, 0)
    high_yellow = (30, 255, 255)

    low_gray = (0, 0, 0)
    high_gray = (179, 255, 233)

    # create masks
    yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
    gray_mask = cv2.inRange(hsv, low_gray, high_gray)

    # combine masks
    combined_mask = cv2.bitwise_or(yellow_mask, gray_mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)
    #cv2.imshow('CM', combined_mask)
    #cv2.waitKey(0)
    # findcontours
    contours, hier = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = combined_mask.copy()

    # draw the outline of all contours
    for cnt in contours:
        cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
    #cv2.imshow('Draw contours of only buildings', image)
    #cv2.waitKey(0)

    inside,hb = check_shop_inside_building(image, midpoint,contours)
    if inside > 0:
        b_dist = detect_nearestbuilding(image, hb,contours,midpoint)

    return b_dist


#image = cv2.imread('Images_all/tmp_1.png')
#min_building_dist = get_building_contours(image,midpoint(image))
#print(min_building_dist)
