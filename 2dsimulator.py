import cv2
import sys
import os
import argparse
import numpy as np
import random
import math

WIDTH = 1000
HEIGHT = 1000


def example_draw():
    Scene = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255

    # Draw rectangle
    cv2.rectangle(Scene, (100, 100), (200, 200), (0, 255, 0), cv2.FILLED)

    # Draw circle
    cv2.circle(Scene, (500, 500), 50, (0, 0, 255), lineType=cv2.LINE_4, thickness=1)

    for i in range(4):
        for j in range(4):
            cv2.circle(Scene, (500+i*200, 500+j*200), 50, (0, 0, 255), cv2.FILLED)
    return Scene

def tablechair_symx():
    """Arrangement of 1 table and 4 chairs that is symmetric across the x axis"""
    Scene = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255
    # Table
    table_tl= (300,430)
    table_br = (700,570)
    cv2.rectangle(Scene, table_tl, table_br, (0, 255, 0), cv2.FILLED)
    radius = 50
    numpairs = 3
    dist_chairs = 20 # distance between the circumference
    offsets_x = np.random.randint(0, 25, size=numpairs)
    offsets_y = np.random.randint(-0.8*20, 30, size=numpairs)
    # Symmetric chairs
    offsetted_circlecenter = table_tl[0]+radius
    for j in range(numpairs): # number of pairs
        if j == 0:
            offsetted_circlecenter += offsets_x[j]
        else:
            offsetted_circlecenter += 2*radius + dist_chairs + offsets_x[j]
        cv2.circle(Scene, (offsetted_circlecenter, table_tl[1]-radius-dist_chairs-offsets_y[j]), radius, (0, 0, 255), cv2.FILLED)
        cv2.circle(Scene, (offsetted_circlecenter, table_br[1]+radius+dist_chairs+offsets_y[j]), radius, (0, 0, 255), cv2.FILLED)
    return Scene


def tablechair_symx_rot():
    """Arrangement of 1 table and 4 chairs that is symmetric across the x axis"""
    Scene = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255
    # Table
    table_tl= (300,430)
    table_br = (700,570)
    table_tr = (table_br[0], table_tl[1])
    table_bl = (table_tl[0], table_br[1])
    radius = 50
    numpairs = 3
    dist_chairs = 20 # distance between the circumference
    offsets_x = np.random.randint(0, 25, size=numpairs)
    offsets_y = np.random.randint(-0.6*20, 30, size=numpairs)
    degree = np.random.randint(0, 180)
    print("# degree =", degree)

    # cv2.rectangle(Scene, table_tl, table_br, (0, 255, 0), cv2.FILLED)
    pts = np.array([[rot_degree(degree, table_tl[0], table_tl[1]), rot_degree(degree, table_tr[0], table_tr[1]), \
           rot_degree(degree, table_br[0], table_br[1]), rot_degree(degree, table_bl[0], table_bl[1])]])
    cv2.fillPoly(Scene, pts, (0, 255, 0))
    offsetted_circlecenter = table_tl[0]+radius
    for j in range(numpairs): # number of pairs
        if j == 0:
            offsetted_circlecenter += offsets_x[j]
        else:
            offsetted_circlecenter += 2*radius + dist_chairs + offsets_x[j]
        cv2.circle(Scene, tuple(rot_degree(degree, offsetted_circlecenter, table_tl[1]-radius-dist_chairs-offsets_y[j])), 
            radius, (0, 0, 255), cv2.FILLED)
        cv2.circle(Scene, tuple(rot_degree(degree, offsetted_circlecenter, table_br[1]+radius+dist_chairs+offsets_y[j])), 
            radius, (0, 0, 255), cv2.FILLED)
    return Scene


def gen_clean_tablechair(rot=False):
    """Arrangement of 1 table and 4 chairs that is symmetric across the x axis"""
    Scene = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255
    # Table
    table_w, table_h = random.randint(400, 800) , 140
    table_tl= ((WIDTH-table_w)/2,(HEIGHT-table_h)/2)# 300, 430
    table_br = (table_tl[0]+table_w,table_tl[1]+table_h)
    table_tr = (table_br[0], table_tl[1])
    table_bl = (table_tl[0], table_br[1])
    radius = random.randint(30,50)
    mindist_chairs = 20 # distance between the circumference
    maxnumpairs = math.floor((table_w + mindist_chairs)/(2*radius + mindist_chairs))
    # minnumpairs = math.ceil(table_w/(2*radius + mindist_chairs + radius*3)) # radius*3 is the max distance between chairs
    minnumpairs = math.ceil(table_w/(radius*2) *0.4)
    numpairs = random.randint(minnumpairs,maxnumpairs) #1-3
    offsets_x = np.random.random(size=numpairs+1)
    offsets_x_exp = [math.exp(e) for e in offsets_x]
    offsets_x = np.array([e/sum(offsets_x_exp) for e in offsets_x_exp])* (table_w - numpairs*2*radius - (numpairs-1)*mindist_chairs) # softmax
    offsets_y = np.random.randint(-0.5*mindist_chairs, 0.5*mindist_chairs, size=numpairs)
    degree = np.random.randint(0, 180) if rot else 0
    pts = np.array([[rot_degree(degree, table_tl[0], table_tl[1]), rot_degree(degree, table_tr[0], table_tr[1]), \
                     rot_degree(degree, table_br[0], table_br[1]), rot_degree(degree, table_bl[0], table_bl[1])]])
    cv2.fillPoly(Scene, pts, (0, 255, 0))
    offsetted_circlecenter = table_tl[0]+radius+offsets_x[0]
    for j in range(numpairs): # number of pairs
        if j > 0:
            offsetted_circlecenter += 2*radius + mindist_chairs + offsets_x[j]
        cv2.circle(Scene, tuple(rot_degree(degree, offsetted_circlecenter, table_tl[1]-radius-mindist_chairs-offsets_y[j])), 
            radius, (0, 0, 255), cv2.FILLED)
        cv2.circle(Scene, tuple(rot_degree(degree, offsetted_circlecenter, table_br[1]+radius+mindist_chairs+offsets_y[j])), 
            radius, (0, 0, 255), cv2.FILLED)
    return Scene

def rot_degree(degree, x, y, origin=[WIDTH/2, HEIGHT/2]):
    """Rotate x and y by degree in counterclockwise direction with center of rotation being origin. Returns numpy array."""
    # print(x, y)
    theta = np.radians(degree)
    R = np.array(( (np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)) ))
    rot_xy = np.dot(R,np.array((x-origin[0],y-origin[1])))
    rot_xy = [int(rot_xy[0].item()+origin[0]), int(rot_xy[1].item()+origin[1])]
    return rot_xy

def gen_data_sample():
    Scene = gen_clean_tablechair()
    cv2.namedWindow('Scene')
    while True:
        cv2.imshow('Scene', Scene)
        Key = cv2.waitKey(-1)
        if Key == ord('s'):
            cv2.imwrite('scene.png', Scene)
        if Key == 27:
            break

if __name__ == '__main__':
    gen_data_sample()