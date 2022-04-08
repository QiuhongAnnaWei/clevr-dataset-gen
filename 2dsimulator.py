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
    # print("# degree =", degree)

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


def rot_degree(degree, x, y, origin=[WIDTH/2, HEIGHT/2]):
    """Rotate x and y by degree in counterclockwise direction with center of rotation being origin. Returns numpy array."""
    # print(x, y)
    theta = np.radians(degree)
    R = np.array(( (np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)) ))
    rot_xy = np.dot(R,np.array((x-origin[0],y-origin[1])))
    rot_xy = [int(rot_xy[0].item()+origin[0]), int(rot_xy[1].item()+origin[1])]
    return rot_xy

def gen_data_sample():
    Scene, centers = gen_tablechair()
    show_image(Scene)

def show_image(Scene):
    cv2.namedWindow('Scene')
    while True:
        cv2.imshow('Scene', Scene)
        Key = cv2.waitKey(-1)
        if Key == ord('s'):
            cv2.imwrite('scene.png', Scene)
        if Key == 27:
            break

def chairoffset_2_center(table_center, table_w, table_h, offsets_x, offsets_y, radius, mindist):
    """Returns (numObj x 2) array describing x and y coordinates of objects topleft -> bottomright in row by row order"""
    numPairs = offsets_x.shape[0]-1 # assume even for now
    centers = np.zeros((2*numPairs, 2))
    table_top, table_bottom = table_center[1]-table_h/2, table_center[1]+table_h/2
    centers[0] = [table_center[0]-table_w/2+radius+offsets_x[0], table_top-radius-mindist-offsets_y[0]] # row 1
    centers[numPairs] = [centers[0,0], table_bottom+radius+mindist+offsets_y[0]] # row 2
    for pair_i in range(1, numPairs):
        centers[pair_i] = [(centers[pair_i-1, 0]+2*radius+mindist+offsets_x[pair_i]), (table_top-radius-mindist-offsets_y[pair_i])]
        centers[pair_i+numPairs] = [centers[pair_i, 0], table_bottom+radius+mindist+offsets_y[pair_i]]
    return centers

def tablecenter_2_pt(center, table_w, table_h):
    table_pts = np.array([[center[0]-table_w/2, center[1]-table_h/2], 
                          [center[0]+table_w/2, center[1]-table_h/2], \
                          [center[0]+table_w/2, center[1]+table_h/2],
                          [center[0]-table_w/2, center[1]+table_h/2]], dtype=np.int32)
    return table_pts  

def build_scene(centers, table_w, table_h, radius, degree):
    """All given centers (np array) are not yet rotated."""
    Scene = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255
    table_pts = tablecenter_2_pt(centers[0, :], table_w, table_h)
    table_pts = np.array([rot_degree(degree, p[0], p[1]) for p in table_pts], dtype=np.int32)
    cv2.fillPoly(Scene, [table_pts], (0, 255, 0))
    for c in centers[1:]:
        cv2.circle(Scene, tuple(rot_degree(degree, c[0], c[1])), radius, (0, 0, 255), cv2.FILLED)
    return Scene

def gen_tablechair(rot=False, to_perturb=True):
    """Arrangement of 1 table and 4 chairs that is symmetric across the x axis"""
    table_w, table_h = random.randint(400, 800), 140
    table_center = [[int(WIDTH/2), int(HEIGHT/2)]] # 1x2 (center of rotation)

    radius = random.randint(30,50)
    mindist_chairs = 20 # distance between the circumference
    maxnumpairs = math.floor((table_w + mindist_chairs)/(2*radius + mindist_chairs))
    # minnumpairs = math.ceil(table_w/(2*radius + mindist_chairs + radius*3)) # radius*3 is the max distance between chairs
    minnumpairs = math.ceil(table_w/(radius*2) *0.4)
    numpairs = random.randint(minnumpairs,maxnumpairs) #1-3
    degree = np.random.randint(0, 180) if rot else 0

    offsets_x = np.random.random(size=numpairs+1)
    offsets_x_exp = [math.exp(e) for e in offsets_x]
    offsets_x = np.array([e/sum(offsets_x_exp) for e in offsets_x_exp])* (table_w - numpairs*2*radius - (numpairs-1)*mindist_chairs) # softmax
    offsets_y = np.random.randint(-0.5*mindist_chairs, 0.5*mindist_chairs, size=numpairs)
    chair_centers = chairoffset_2_center(table_center[0], table_w, table_h, offsets_x, offsets_y, radius, mindist_chairs) # list
    centers = np.concatenate((np.array(table_center), np.array(chair_centers)), axis=0) # nx2
    Scene = build_scene(centers, table_w, table_h, radius, degree)
    if to_perturb:
        per_Scene, per_centers = perturb(centers, table_w, table_h, radius, degree)
    return Scene, centers, per_Scene, per_centers


def perturb(centers, table_w, table_h, radius, degree=0):
    """All given centers (np array) are not yet rotated."""
    # translation perturbation (applied to center of table and whole scene)
    centers += np.random.normal(0,30,2)
    random_indices = list(range(1, centers.shape[0]-1))
    random.shuffle(random_indices) # in-place
    for chair_i in random_indices:
        new_chair_cen = centers[chair_i] + np.random.normal(0,30,2)
        while (not is_valid(centers, chair_i, new_chair_cen, radius, table_w, table_h)):
            new_chair_cen = centers[chair_i] + np.random.normal(0,30,2)
        centers[chair_i] = new_chair_cen
    # np.random.normal(0,1,(3,2))
    centers = np.array([rot_degree(degree, c[0], c[1]) for c in centers], dtype=np.int32)
    Scene = build_scene(centers, table_w, table_h, radius, degree)
    return Scene, centers

def is_valid(centers, chair_i, new_chair_cen, radius, table_w, table_h):
    """checks if new_chair_cen for the ith chair is valid: no overlap with any other object"""
    # check against table
    if (abs(centers[0,0]-new_chair_cen[0])<radius+table_w/2) and (abs(centers[0,1]-new_chair_cen[1])<radius+table_h/2):
        return False
    # check against other chairs
    for i in range(1, centers.shape[0]):
        if i==chair_i:
            continue
        if np.linalg.norm(new_chair_cen-centers[i]) < 2*radius: # euclidean distance
            return False
    return True

def gen_data(n, rot=False):
    """Generate n data samples:
    1. per_poses, per_shape: n x dim numpy array, where ith row describes pose/shape of the ith perturbed scene. (1st row=table)
    2. clean_poses, clean_shape: n x dim numpy array, where ith row describes pose/shape of the ith clean scene (corresponds to ith perturbed scene)
    3. Images of clean and pertubed scene for reference
    """
    pose_d = 3 # center, rotation
    shape_d = 2
    # Chair: center, -1 (rotation) | type, radius (fixed), radius (fixed)
    # Table: center, rotation,     | type, width (fixed),  height (fixed)
    clean_poses, per_poses = np.zeros((n,pose_d)), np.zeros((n,pose_d))
    clean_shape, per_shape = np.zeros((n,shape_d)),  np.zeros((n,shape_d))

    Scene, centers, per_Scene, per_centers = gen_tablechair(rot=rot, to_perturb=True)
    # clean_poses[:,0:2] = centers # flatten
    # per_poses[:,0:2] = per_centers
    show_image(Scene)
    show_image(per_Scene)
    # TODO: file save and image write (cv2.imwrite('scene.png', Scene))
    # TODO: poses: flatten + rotation
    # if rot:
    #     pass 
    # else:
    #     clean_poses[0,2] = 0 # table's rotation is 0
    #     clean_poses[:,2] = -1 # chairs' rotation is -1
    #     per_poses[0,2] = 0 # table's rotation is 0
    #     per_poses[:,2] = -1 # chairs' rotation is -1
    print("clean_poses:\n", clean_poses)
    print("per_poses:\n", per_poses)

    # TODO: clean_shape, per_shape


if __name__ == '__main__':
    gen_data(1)