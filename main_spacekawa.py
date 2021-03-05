import building,road,park
import ntpath
import argparse as ap
import cv2
import pandas as pd
import glob
import sys
import os


def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def midpoint(image):
    a = image.shape
    y = int((a[0])/2)
    x = int((a[1])/2)
    return (x,y)


def main():
    data = pd.DataFrame(columns=['Filename', 'Distance', 'Flag'])



    arg = ap.ArgumentParser()
    arg.add_argument("--imagefolder", required=True,help="Give the folder path to input images")
    args = vars(arg.parse_args())

    files = glob.glob(args["imagefolder"])
    for file in files:
        distances_list = []
        FLAG = ''
        DISTANCE = 0
        name = get_filename(file)[:-4]
        path = os.path.join('Images_all', get_filename(file))
        #print('path', path)
    # image is read into memory
        img = cv2.imread(path)
        #print(args["image"])park_dist
        #cv2.imshow('Original image :',img)
        #cv2.waitKey(0)

        shop_location = (midpoint(img))
        min_building_distance = building.get_building_contours(img,shop_location)
        min_road_distance = road.get_road_contours(img,shop_location)
        min_park_distance = park.get_park_contours(img,shop_location)

        #print('Distance of the shop from the road -', min_road_distance)
        #print('Distance of the shop from the building -', min_building_distance)
        #print('Distance of the shop from the park -', min_park_distance)

        distances_list.append(min_road_distance)
        distances_list.append(min_building_distance)
        distances_list.append(min_park_distance)

        DISTANCE = min(n for n in distances_list  if n>0)
        if min_road_distance == DISTANCE:
            FLAG = 'ROAD'
        elif min_building_distance == DISTANCE:
            FLAG = 'BUILDING'
        else:
            FLAG = 'PARK'
        data = data.append({'Filename': path, 'Distance': DISTANCE, 'Flag': FLAG}, ignore_index=True)

    print('###########################################')
    print(data)
    print('###########################################')
    cv2.destroyAllWindows()





main()
